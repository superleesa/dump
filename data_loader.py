from datasets import load_dataset


def generate_prompt(data_record):
    if data_record["input"]:
        result = f'### 指示:{data_record["instruction"]}\n### 入力:{data_record["output"]}\n回答:{data_record["output"]}'
    else:
        result = f'### 指示:{data_record["instruction"]}\n回答:{data_record["output"]}'

    data_record["text"] = result  # mutating
    return data_record


def load_my_dataset(cutoff_length=512):
    dataset_name = "kunishou/databricks-dolly-15k-ja"
    dataset = load_dataset(dataset_name)
    dataset = dataset["train"].map(generate_prompt)
    dataset = dataset["train"].filter(lambda row: len(row["text"]) < cutoff_length).shuffle()
    dataset = dataset["train"].train_test_split(test_size=2000)
    train_dataset, eval_dataset = dataset["train"], dataset["test"]
    return train_dataset, eval_dataset
    