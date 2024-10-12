from dataclasses import dataclass, field
from typing import List, Optional

from transformers import TrainingArguments


@dataclass
class BaseTrainingArguments:
    run_id: str = field(
        default="llama2_qlora", metadata={"help": "A unique 'name' of this experiment, used to store metadata on the DHT"}
    )
    initial_peers: List[str] = field(
        default_factory=list,
        metadata={
            "help": "Multiaddrs of the peers that will welcome you into the existing collaboration. "
            "Example: /ip4/203.0.113.1/tcp/31337/p2p/XXXX /ip4/203.0.113.2/tcp/7777/p2p/YYYY"
        },
    )
    use_ipfs: bool = field(
        default=False,
        metadata={
            "help": "Use IPFS to find initial_peers. If enabled, you only need to provide /p2p/XXXX part of the multiaddrs "
            "for the initial_peers (no need to specify a particular IPv4/IPv6 host and port)"
        },
    )
    host_maddrs: List[str] = field(
        default_factory=lambda: ["/ip4/0.0.0.0/tcp/0"],
        metadata={
            "help": "Multiaddrs to listen for external connections from other p2p instances. "
            "Defaults to all IPv4 interfaces and the TCP protocol: /ip4/0.0.0.0/tcp/0"
        },
    )
    announce_maddrs: List[str] = field(
        default_factory=list,
        metadata={"help": "Visible multiaddrs the host announces for external connections from other p2p instances"},
    )
    identity_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to a pre-generated private key file. If defined, makes the peer ID deterministic. "
            "If the file does not exist yet, writes a new private key to this file."
        },
    )


@dataclass
class QLoRAArguments:
    lora_r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    lora_alpha: int = field(default=16, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "Lora dropout"})
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"],
        metadata={"help": "List of module names to apply LoRA to"}
    )
    bits: int = field(default=4, metadata={"help": "Quantization bits"})

@dataclass
class Llama2QLoRATrainingArguments(TrainingArguments):
    model_name: str = field(default="meta-llama/Llama-2-7b-hf", metadata={"help": "Model name or path"})
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 32
    num_train_epochs: float = 3.0
    learning_rate: float = 2e-4
    fp16: bool = False
    bf16: bool = True
    max_grad_norm: float = 0.3
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "constant"
    logging_steps: int = 10
    optim: str = "paged_adamw_32bit"
    weight_decay: float = 0.01

@dataclass
class AveragerArguments:
    target_group_size: int = field(default=256, metadata={"help": "Maximum group size for all-reduce"})


@dataclass
class ProgressTrackerArguments:
    min_refresh_period: float = field(
        default=0.5, metadata={"help": "Wait for at least this many seconds before fetching new collaboration state"}
    )
    max_refresh_period: float = field(
        default=30, metadata={"help": "Wait for at most this many seconds before fetching new collaboration state"}
    )
    default_refresh_period: float = field(
        default=3, metadata={"help": "Attempt to fetch collaboration state every this often until successful"}
    )
    expected_drift_peers: float = field(
        default=3, metadata={"help": "Trainer assumes that this many new peers can join per step"}
    )
    expected_drift_rate: float = field(
        default=0.2, metadata={"help": "Trainer assumes that this fraction of current size can join per step"}
    )
    metadata_expiration: float = field(
        default=120, metadata={"help": "Peer's metadata will be removed if not updated in this many seconds"}
    )


@dataclass
class OptimizerArguments:
    target_batch_size: int = field(
        default=4096,
        metadata={"help": "Perform optimizer step after all peers collectively accumulate this many samples"},
    )
    client_mode: bool = field(
        default=False,
        metadata={"help": "Of True, runs training without incoming connections, in a firewall-compatible mode"},
    )
    batch_size_lead: int = field(
        default=0,
        metadata={"help": "Optional: begin looking for group in advance, this many samples before target_batch_size"},
    )
    bandwidth: float = field(
        default=100.0,
        metadata={"help": "Available network bandwidth, in mbps (used for load balancing in all-reduce)"},
    )
    averaging_timeout: float = field(
        default=60.0, metadata={"help": "Give up on averaging step after this many seconds"}
    )
    matchmaking_time: float = field(
        default=5.0, metadata={"help": "When looking for group, wait for requests for at least this many seconds"}
    )


@dataclass
class CollaborationArguments(OptimizerArguments, BaseTrainingArguments, QLoRAArguments):
    statistics_expiration: float = field(
        default=600, metadata={"help": "Statistics will be removed if not updated in this many seconds"}
    )
    backup_every_steps: int = field(
        default=10, metadata={"help": "Frequency of backups to restore from in case of encountering NaN values"}
    )


@dataclass
class DatasetArguments:
    dataset_path: Optional[str] = field(
        default="data/tokenized_arxiv_abstract", metadata={"help": "Path to the tokenized dataset"}
    )
    tokenizer_path: Optional[str] = field(default="data/tokenizer", metadata={"help": "Path to the tokenizer"})
    config_path: Optional[str] = field(
        default="albert-large-v2",
        metadata={"help": "Path to the model config"},
    )
    cache_dir: Optional[str] = field(default="data", metadata={"help": "Path to the cache"})
    