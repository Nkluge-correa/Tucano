# Bash Scripts

Given that our study was performed in the [Marvin cluster](https://www.hpc.uni-bonn.de/en/systems/marvin), which uses SLURM for job scheduling, all major scripts were launched via bash scripts:

- Feature Alignment: [`vitucano-pretraining.sh`](../scripts/vitucano-pretraining.sh).
- Visual Instruction Tuning: [`vitucano-sft.sh`](../scripts/vitucano-sft.sh).

The configuration used to train both ViTucano models can be seen below:

| Argument                     | Value                              |
|------------------------------|------------------------------------|
| `conv_version`               | "llama"                            |
| `vision_tower`               | "google/siglip-so400m-patch14-384" |
| `connector_type`             | "mlp2x_gelu"                       |
| `image_aspect_ratio`         | "square"                           |
| `attn_implementation`        | "flash_attention_2"                |
| `per_device_train_batch_size`| 16                                 |
| `per_device_eval_batch_size` | 4                                  |
| `gradient_accumulation_steps`| 1                                  |
| `learning_rate`              | 2e-5                               |
| `weight_decay`               | 0.0                                |
| `warmup_ratio`               | 0.03                               |
| `lr_scheduler_type`          | "cosine"                           |
| `logging_steps`              | 1                                  |
| `tf32`                       | true                               |
| `bf16`                       | true                               |
| `gradient_checkpointing`     | true                               |
| `dataloader_num_workers`     | 8                                  |
| `lazy_preprocess`            | true                               |
| `tokenizer_use_fast`         | true                               |
