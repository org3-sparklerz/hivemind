#!/usr/bin/env python3

""" This script builds a pre-tokenized compressed representation of ArXiv abstracts using huggingface/datasets """

import random
from functools import partial

import nltk
from datasets import load_dataset
from transformers import AutoTokenizer

# Define column names for the tokenized dataset
COLUMN_NAMES = ("attention_mask", "input_ids", "special_tokens_mask")

def create_instances_from_document(tokenizer, document, max_seq_length):
    """
    Creates training instances from a single document.
    This function tokenizes the input text and ensures it fits within the specified maximum sequence length.
    """
    instances = []
    current_chunk = []
    current_length = 0

    # Tokenize sentences in the document
    segmented_sents = list(nltk.sent_tokenize(document))
    
    for i, sent in enumerate(segmented_sents):
        current_chunk.append(sent)
        current_length += len(tokenizer.tokenize(sent))
        
        if i == len(segmented_sents) - 1 or current_length >= max_seq_length:
            if len(current_chunk) > 0:
                tokens_a = " ".join(current_chunk)
                
                instance = tokenizer(
                    tokens_a,
                    truncation=True,
                    max_length=max_seq_length,
                    return_special_tokens_mask=True,
                )

                assert len(instance["input_ids"]) <= max_seq_length
                instances.append(instance)
            
            current_chunk = []
            current_length = 0

    return instances

def tokenize_function(tokenizer, examples):
    # Remove empty texts
    texts = (text for text in examples["abstract"] if len(text) > 0 and not text.isspace())
    
    new_examples = {col: [] for col in COLUMN_NAMES}
    
    for text in texts:
        instances = create_instances_from_document(tokenizer, text, max_seq_length=512)
        
        for instance in instances:
            for key, value in instance.items():
                new_examples[key].append(value)
    
    return new_examples

if __name__ == "__main__":
    random.seed(0)
    nltk.download("punkt")
    
    # Load the LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained("llama-tokenizer")  # Update with your specific tokenizer path or identifier
    
    # Load the ArXiv abstract dataset
    arxiv_dataset = load_dataset("ash001/arxiv-abstract", split="train", cache_dir="./data/cache")
    
    # Tokenize the dataset
    tokenized_datasets = arxiv_dataset.map(
        partial(tokenize_function, tokenizer),
        batched=True,
        num_proc=8,
        remove_columns=["abstract"],  # Remove original abstract column after tokenization
    )
    
    # Save the tokenized dataset to disk
    tokenized_datasets.save_to_disk("./data/tokenized_arxiv_abstract")
    
    # Save the tokenizer configuration
    tokenizer.save_pretrained("./data/tokenizer")