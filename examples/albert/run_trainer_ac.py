#!/usr/bin/env python3

import os
import sys
from dataclasses import asdict
from pathlib import Path

import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

import hivemind
from hivemind.optim.state_averager import TrainingStateAverager
from hivemind.utils.logging import get_logger, use_hivemind_log_handler
from hivemind.utils.networking import log_visible_maddrs

import utils
from arguments_ac import CollaborationArguments, Llama2QLoRATrainingArguments

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__name__)

def setup_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        device_map="auto",
        load_in_4bit=args.bits == 4,
        load_in_8bit=args.bits == 8,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    return model, tokenizer

def main():
    parser = HfArgumentParser(
        (
            CollaborationArguments, 
            Llama2QLoRATrainingArguments
            )
    )
    collaboration_args, training_args = parser.parse_args_into_dataclasses()

    logger.info(f"Found {len(collaboration_args.initial_peers)} initial peers: {collaboration_args.initial_peers}")

    dht = hivemind.DHT(
        start=True,
        initial_peers=collaboration_args.initial_peers,
        client_mode=collaboration_args.client_mode,
        host_maddrs=collaboration_args.host_maddrs,
        announce_maddrs=collaboration_args.announce_maddrs,
    )
    log_visible_maddrs(dht.get_visible_maddrs())

    model, tokenizer = setup_model_and_tokenizer(training_args)

    # Load dataset
    dataset = load_dataset("ash001/arxiv-abstract", split="train")
        
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Initialize averager
    averager = TrainingStateAverager(
        dht=dht,
        optimizer=model.parameters(),
        scheduler=None,  # We're not using a scheduler in this example
        prefix=collaboration_args.run_id,
        state_compression=hivemind.Float16Compression(),
        bandwidth=collaboration_args.bandwidth,
        client_mode=collaboration_args.client_mode,
        start=True,
        **asdict(collaboration_args),
    )

    # Initialize Trainer
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Train
    trainer.train()

    # Save the final model
    trainer.save_model()

if __name__ == "__main__":
    main()
