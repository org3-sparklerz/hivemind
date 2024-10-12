#!/usr/bin/env python3

import random
from functools import partial
from datasets import load_dataset
from transformers import AutoTokenizer

COLUMN_NAMES = ("input_ids", "attention_mask")

def tokenize_function(tokenizer, examples):
    tokenized_examples = tokenizer(
        examples["abstract"],
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt",
    )
    return {
        "input_ids": tokenized_examples["input_ids"][0],
        "attention_mask": tokenized_examples["attention_mask"][0],
    }

def main():
    random.seed(0)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    arxiv_abstract = load_dataset("ash001/arxiv-abstract", cache_dir="./data/cache")

    tokenized_datasets = arxiv_abstract.map(
        partial(tokenize_function, tokenizer),
        batched=True,
        num_proc=8,
        remove_columns=arxiv_abstract["train"].column_names,
    )

    tokenized_datasets.save_to_disk("./data/tokenized_arxiv_abstract")
    tokenizer.save_pretrained("./data/tokenizer")

if __name__ == "__main__":
    main()