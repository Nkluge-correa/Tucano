from typing import Optional, Union
from dataclasses import dataclass, field

@dataclass
class TrainingArguments:
    """Class to hold the training arguments."""
    
    ################### Directory settings ###################
    checkpoint_dir: Optional[str] = field(
        default="./checkpoints",
        metadata={"help": "The directory to save the model checkpoints."},
    )
    train_dataset_dir: Optional[str] = field(
        default="./gigaverbo/train",
        metadata={"help": "The directory where the dataset is stored."},
    )
    val_dataset_dir: Optional[str] = field(
        default="./gigaverbo/val",
        metadata={"help": "The directory where the validation dataset is stored."},
    )
    cache_dir: Optional[str] = field(
        default="./cache",
        metadata={"help": "The directory to save the cache files."},
    )

    ################### Data loading settings ###################
    num_workers_for_dataloader: Optional[int] = field(
        default=4,
        metadata={"help": "The number of workers for the dataloader."},
    )
    pin_memory: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to pin the memory for faster data transfer on the dataloader."},
    )

    ################### Model architecture settings ###################
    vocab_size: Optional[int] = field(
        default=32000,
        metadata={"help": "The vocab size of the tokenizer."},
    )
    num_hidden_layers: Optional[int] = field(
        default=12,
        metadata={"help": "The number of layers in the model."},
    )
    num_attention_heads: Optional[int] = field(
        default=12,
        metadata={"help": "The number of attention heads in the model."},
    )
    num_key_value_heads: Optional[int] = field(
        default=12,
        metadata={"help": "The number of key-value heads for group query attention."},
    )
    hidden_size: Optional[int] = field(
        default=768,
        metadata={"help": "The embedding size of the model."},
    )
    intermediate_size: Optional[int] = field(
        default=None,
        metadata={"help": "The intermediate size of the model. Defaults to 4 * hidden_size."},
    )
    max_position_embeddings: Optional[int] = field(
        default=2048,
        metadata={"help": "The maximum context size of the model."},
    )
    hidden_act: Optional[str] = field(
        default="silu",
        metadata={"help": "The activation function to use."},
    )
    output_hidden_states: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to output the hidden states."},
    )
    use_cache: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use cache for faster inference."},
    )

    ################### Training settings ###################
    total_batch_size: Optional[int] = field(
        default=524288,
        metadata={"help": "The total batch size in tokens."},
    )
    micro_batch_size: Optional[int] = field(
        default=32,
        metadata={"help": "The micro batch size."},
    )
    eval_micro_batch_size: Optional[int] = field(
        default=32,
        metadata={"help": "The evaluation micro batch size."},
    )
    num_train_epochs: Optional[Union[float, int]] = field(
        default=1,
        metadata={"help": "The number of training epochs."},
    )
    warmup_steps: Optional[int] = field(
        default=1000,
        metadata={"help": "The number of warmup steps."},
    )
    max_learning_rate: Optional[float] = field(
        default=1e-3,
        metadata={"help": "The initial maximum learning rate."},
    )
    min_learning_rate: Optional[float] = field(
        default=1e-4,
        metadata={"help": "The minimum learning rate."},
    )
    weight_decay: Optional[float] = field(
        default=0.1,
        metadata={"help": "The weight decay to apply."},
    )
    beta1: Optional[float] = field(
        default=0.9,
        metadata={"help": "The beta1 parameter for the Adam optimizer."},
    )
    beta2: Optional[float] = field(
        default=0.95,
        metadata={"help": "The beta2 parameter for the Adam optimizer."},
    )
    eps: Optional[float] = field(
        default=1e-8,
        metadata={"help": "The epsilon parameter for the Adam optimizer."},
    )
    lr_decay_iters_coef: Optional[float] = field(
        default=0.9,
        metadata={"help": "We will decay the learning rate until 90% of the total steps."},
    )
    seed: Optional[int] = field(
        default=1337,
        metadata={"help": "The seed for PyTorch to ensure reproducibility."},
    )

    ################### Precision and optimization settings ###################
    mat_mul_precision: Optional[str] = field(
        default="highest",
        metadata={"help": (
            "The precision for matrix multiplication. "
            "Options: highest, high, medium."
        )}
    )
    tf32: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use tf32 mode (requires Ampere GPU)."},
    )
    bf16: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use bf16 mode."},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use gradient checkpointing."},
    )

    ################### Hub settings ###################
    push_to_hub: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to push the model to the hub."},
    )
    hub_token: Optional[str] = field(
        default=None,
        metadata={"help": "The token to the huggingface hub."},
    )
    hub_model_id: Optional[str] = field(
        default=None,
        metadata={"help": "The model id to push to the hub (e.g., userName/modelName)."},
    )

    ################### Tokenizer settings ###################
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the tokenizer to use."},
    )

    ################### Checkpoint settings ###################
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "The path to the checkpoint to resume from."},
    )

    checkpointing_steps: Optional[int] = field(
        default=2000,
        metadata={"help": "The number of steps to save a checkpoint. Eval will be performed after each checkpoint."},
    )

    ################### Miscellaneous settings ###################
    sanity_check: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run a sanity check on a small dataset."},
    )
    wandb_token: Optional[str] = field(
        default=None,
        metadata={"help": "The token to your W&B account."},
    )
    wandb_id: Optional[str] = field(
        default=None,
        metadata={"help": "The id of the W&B run."},
    )
