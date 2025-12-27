import torch
from datasets import load_dataset
from transformers import AutoTokenizer
import config


# -------------------------
# Tokenizer
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

config.VOCAB_SIZE = len(tokenizer)


# -------------------------
# Dataset
# -------------------------
dataset = load_dataset(
    config.DATASET_NAME,
    config.DATASET_CONFIG
)


def tokenize(example):
    text = example[config.TEXT_FIELD]

    # Skip empty lines explicitly
    if not text or text.strip() == "":
        return {"ids": []}

    return {
        "ids": tokenizer(
            text,
            truncation=False,
            add_special_tokens=False
        )["input_ids"]
    }


dataset = dataset.map(
    tokenize,
    remove_columns=[config.TEXT_FIELD],
)


# -------------------------
# Flatten token stream
# -------------------------
def flatten(split):
    ids = []
    for row in dataset[split]["ids"]:
        ids.extend(row)

    # Keep on CPU — training code moves batches to device
    return torch.tensor(ids, dtype=torch.long, device="cpu")


train_ids = flatten("train")
valid_ids = flatten("validation")
