from transformers import (
                    Trainer,
                    AutoTokenizer,
                    AutoModelForCausalLM,
                    TrainingArguments,
                    EarlyStoppingCallback,
                    BitsAndBytesConfig,
                    DataCollatorForLanguageModeling
                )

import torch
from data_loader import load_my_dataset
import random

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import json
import os
from accelerate import Accelerator

import torch._dynamo



def main():
    # suppress dynamo errors (happens with older gpus)
    torch._dynamo.config.suppress_errors = True
    
    ### params ###
    CUTOFF_LENGTH = 512
    model_name = "tokyotech-llm/Swallow-7b-instruct-hf"
    peft_name = "sft-swallow-13"
    output_dir = "v2-sft-swallow-13-result"
    ##############

    def tokenize(prompt, tokenizer):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=CUTOFF_LENGTH,
            padding=False,
        )
    
        return {
                "input_ids": result["input_ids"],
                "attention_mask": result["attention_mask"],
            }

    train_data, eval_data = load_my_dataset(cutoff_length=CUTOFF_LENGTH)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    train_data = train_data.map(lambda x: tokenize(x["text"], tokenizer))
    eval_data = eval_data.map(lambda x: tokenize(x["text"], tokenizer))

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # see: https://github.com/huggingface/accelerate/issues/1840#issuecomment-1683105994
    device_index = Accelerator().process_index
    device_map = {"": device_index}

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map, quantization_config=bnb_config)

    lora_config = LoraConfig(
    r=32,  # the rank of the matrix = # of columns
    lora_alpha=64,  # how important the lora matrix (scaling factor)?
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    lora_dropout=0,
    bias="none",
    task_type="CAUSAL_LM"
    )

    # model.gradient_checkpointing_enable()
    quantized_model = prepare_model_for_kbit_training(model)
    
    peft_model = get_peft_model(quantized_model, lora_config)

    save_strategy_config = {
        "evaluation_strategy": "steps",
        "save_strategy": "steps",
        "eval_steps": 100,
        "save_steps": 100,
        "logging_steps": 100,
        "output_dir": output_dir,
        "save_total_limit": 100,
        "load_best_model_at_end": True
    }


    training_arguments = TrainingArguments(
        prediction_loss_only=True,
        num_train_epochs=10,
        optim="adamw_torch",
        learning_rate=3e-4,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=10e-5,
        weight_decay=0.1,
        max_grad_norm=1,
        torch_compile=True,
        lr_scheduler_type="cosine",
        warmup_steps=500,
        # per_gpu_train_batch_size=64,
        # per_gpu_eval_batch_size=64,
        auto_find_batch_size=True,
        neftune_noise_alpha=5,
        **save_strategy_config
    )

    trainer = Trainer(
        peft_model,
        train_dataset=train_data,
        eval_dataset= eval_data,
        args=training_arguments,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    peft_model.config.use_cache = False
    print("training starts....")
    try:
        trainer.train(resume_from_checkpoint=True)
    except ValueError:
        trainer.train()
    except FileNotFoundError:
        trainer.train()
        
    
    peft_model.config.use_cache = True

    trainer.model.save_pretrained(peft_name)
    with open("sft-swallow_log_history.json", "w") as f:
        json.dump(trainer.state.log_history, f)


if __name__ == "__main__":
    main()
