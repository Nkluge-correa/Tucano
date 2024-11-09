# Configuring the `yaml` specifications file

All arguments passed to the [`train-tucano.py`](./train-tucano.py) are done via a specification file. Here are all the arguments they carry:

| **Argument**                    | **Description**                                                                                      |
|---------------------------------|------------------------------------------------------------------------------------------------------|
| `checkpoint_dir`                | The directory to save the model checkpoints.                                                         |
| `train_dataset_dir`             | The directory where the training dataset is stored.                                                  |
| `val_dataset_dir`               | The directory where the validation dataset is stored.                                                |
| `cache_dir`                     | The directory to save the cache files.                                                               |
| `num_workers_for_dataloader`    | The number of workers for the data loader.                                                           |
| `pin_memory`                    | Whether to pin memory for faster data transfer in the data loader.                                   |
| `vocab_size`                    | The vocabulary size of the tokenizer.                                                                |
| `num_hidden_layers`             | The number of layers in the model.                                                                   |
| `num_attention_heads`           | The number of attention heads in the model.                                                          |
| `num_key_value_heads`           | The number of key-value heads for group query attention.                                             |
| `hidden_size`                   | The embedding size of the model.                                                                     |
| `intermediate_size`             | The intermediate size of the model, defaulting to 4 * hidden_size.                                   |
| `max_position_embeddings`       | The maximum context size of the model.                                                               |
| `hidden_act`                    | The activation function to use (e.g., "silu").                                                       |
| `output_hidden_states`          | Whether to output the hidden states.                                                                 |
| `use_cache`                     | Whether to use cache for faster inference.                                                           |
| `total_batch_size`              | The total batch size in tokens.                                                                      |
| `micro_batch_size`              | The micro batch size.                                                                                |
| `eval_micro_batch_size`         | The evaluation micro batch size.                                                                     |
| `num_train_epochs`              | The number of training epochs.                                                                       |
| `warmup_steps`                  | The number of warmup steps.                                                                          |
| `max_learning_rate`             | The initial maximum learning rate.                                                                   |
| `min_learning_rate`             | The minimum learning rate.                                                                           |
| `weight_decay`                  | The weight decay to apply during training.                                                           |
| `beta1`                         | The beta1 parameter for the Adam optimizer.                                                          |
| `beta2`                         | The beta2 parameter for the Adam optimizer.                                                          |
| `eps`                           | The epsilon parameter for the Adam optimizer.                                                        |
| `lr_decay_iters_coef`           | Coefficient for decaying the learning rate until 90% of total steps.                                 |
| `seed`                          | The random seed for PyTorch to ensure reproducibility.                                               |
| `mat_mul_precision`             | The precision for matrix multiplication. Options: highest, high, medium.                             |
| `tf32`                          | Whether to use TensorFloat-32 mode (requires Ampere GPU).                                            |
| `bf16`                          | Whether to use bfloat16 mode.                                                                        |
| `gradient_checkpointing`        | Whether to use gradient checkpointing to save memory.                                                |
| `push_to_hub`                   | Whether to push the model to the Hugging Face hub.                                                   |
| `hub_token`                     | The authentication token for the Hugging Face hub.                                                   |
| `hub_model_id`                  | The model ID to push to the Hugging Face hub (e.g., userName/modelName).                             |
| `tokenizer_name`                | The name of the tokenizer to use.                                                                    |
| `resume_from_checkpoint`        | The path to the checkpoint to resume training from.                                                  |
| `checkpointing_steps`           | The number of steps to save a checkpoint, performing evaluation after each checkpoint.               |
| `sanity_check`                  | Whether to run a sanity check on a small dataset.                                                    |
| `wandb_token`                   | The authentication token for the Weights & Biases account.                                           |
| `wandb_id`                      | The ID of the Weights & Biases run.                                                                  |

To learn more, please see the structure of the available configuration files (e.g., [specs-tucano-1b1.yml](./specs-tucano-1b1.yml)) and modify them to your own needs. You can also modify directly the [train-tucano.py](./train-tucano.py) or [specifications.py](./specifications.py) scripts.
