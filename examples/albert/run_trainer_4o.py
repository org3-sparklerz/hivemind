#!/usr/bin/env python3

import os
import pickle
import sys
from dataclasses import asdict
from pathlib import Path

import torch
import transformers
from datasets import load_from_disk
from torch.utils.data import DataLoader

from transformers import DataCollatorForLanguageModeling, HfArgumentParser, set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer

from hivemind import DHT, Float16Compression, Optimizer, get_dht_time
from hivemind.optim.state_averager import LRSchedulerBase
from hivemind.utils.logging import get_logger, use_hivemind_log_handler
from hivemind.utils.networking import log_visible_maddrs

import utils
from arguments_4o import (
   QLoRATrainingArguments,
   AveragerArguments,
   CollaborationArguments,
   DatasetArguments,
   ProgressTrackerArguments,
)

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__name__)


def setup_transformers_logging(process_rank: int):
   if is_main_process(process_rank):
       transformers.utils.logging.set_verbosity_info()
       transformers.utils.logging.disable_default_handler()
       transformers.utils.logging.enable_propagation()


def get_model(training_args):
   logger.info(f"Loading LLaMA model")   
   model_name_or_path = 'llama-7b' # Update with your specific model path or identifier.   
   model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    return model

def main():
   parser = HfArgumentParser(
        (
            QLoRATrainingArguments,
            DatasetArguments,
            CollaborationArguments,
            AveragerArguments,
            ProgressTrackerArguments,
        )
    )
   training_args, dataset_args, collaboration_args, averager_args, tracker_args = parser.parse_args_into_dataclasses()
    logger.info(f"Found {len(collaboration_args.initial_peers)} initial peers: {collaboration_args.initial_peers}")
   
   setup_transformers_logging(training_args.local_rank)
   logger.info(f"Training/evaluation parameters:\n{training_args}")

    # Set seed before initializing model.
   set_seed(training_args.seed)
   
   try:
       tokenizer_path_or_name='llama-tokenizer' # Update with your specific tokenizer path or identifier. 
       tokenizer = AutoTokenizer.from_pretrained(tokenizer_path_or_name)
       
       tokenized_datasets = load_from_disk(Path(dataset_args.dataset_path))
       
       data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)
       
       validators, local_public_key = utils.make_validators(collaboration_args.run_id)
       
       dht = DHT(
           start=True,
           initial_peers=collaboration_args.initial_peers,
           client_mode=collaboration_args.client_mode,
           record_validators=validators,
           use_ipfs=collaboration_args.use_ipfs,
           host_maddrs=collaboration_args.host_maddrs,
           announce_maddrs=collaboration_args.announce_maddrs,
           identity_path=collaboration_args.identity_path,
       )
       log_visible_maddrs(dht.get_visible_maddrs(), only_p2p=collaboration_args.use_ipfs)

       total_batch_size_per_step = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
       if torch.cuda.device_count() != 0:
           total_batch_size_per_step *= torch.cuda.device_count()

       adjusted_target_batch_size = collaboration_args.target_batch_size - collaboration_args.batch_size_lead

       # We need to make such a lambda function instead of just an optimizer instance
       # to make hivemind.Optimizer(..., offload_optimizer=True) work
       opt_fn=lambda params : torch.optim.AdamW(params=params)
      
       optimizer_params=[
           {
               'params': [p for n,p in model.named_parameters() if not any(nd in n for nd in ["bias", "LayerNorm.weight"])],
               'weight_decay': training_args.weight_decay},
           {
               'params': [p for n,p in model.named_parameters() if any(nd in n for nd in ["bias", "LayerNorm.weight"])],
               'weight_decay': 0}
       ]

       optimizer=torch.optim.AdamW(optimizer_params)

       scheduler=torch.optim.lr_scheduler.LinearLR(optimizer,start_factor=training_args.learning_rate,end_factor=training_args.learning_rate,num_iters=int(training_args.num_train_epochs))

       optimizer_wrapper=lambda params : Optimizer(dht=dht,opt_fn,opt_params=params,scheduler=scheduler)

      trainer_with_independent_shuffling_class=lambda : TrainerWithIndependentShuffling(model=model,args=training_args,data_collator=data_collator,dataset=train_dataset if training_args.do_train else None,eval_dataset=None if not training_args.do_eval else eval_dataset,opt=(optimizer_wrapper,None),callbacks=[CollaborativeCallback(dht=dht,opt_fn,opt_params=params,scheduler=scheduler)])

      trainer_with_independent_shuffling_class().train()

if __name__=="__main__":
     main()
     