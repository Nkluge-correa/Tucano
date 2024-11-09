import torch
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl import (
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
from liger_kernel.transformers import apply_liger_kernel_to_llama

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, DPOConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()

    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        attn_implementation="sdpa",
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, **model_kwargs
    )

    # Adding this line automatically monkey-patches the model with the optimized Liger kernels, i.e.,
    # a collection of Triton kernels designed specifically for LLM training ([source](https://github.com/linkedin/Liger-Kernel))
    apply_liger_kernel_to_llama()
    
    peft_config = get_peft_config(model_config)
    if peft_config is None:
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, **model_kwargs
        )
    else:
        ref_model = None
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    if script_args.ignore_bias_buffers:
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    dataset = load_dataset(script_args.dataset_name, split="portuguese")
    # Some pre-processing of the dataset
    df = dataset.to_pandas()

    def transform_row(row):
        # Format the 'chosen' column as per the chat structure
        chosen = [
            {"content": row["instruction"], "role": "user"},
            {"content": row["chosen_response"], "role": "assistant"}
        ]
        
        # Format the 'rejected' column in the same way
        rejected = [
            {"content": row["instruction"], "role": "user"},
            {"content": row["rejected_response"], "role": "assistant"}
        ]
        
        return pd.Series([chosen, rejected], index=["chosen", "rejected"])

    df[["chosen", "rejected"]] = df.apply(transform_row, axis=1)
    df = df.drop(columns=["instruction", "chosen_response", "rejected_response"])

    dataset = Dataset.from_pandas(df)
    dataset = dataset.train_test_split(test_size=1000, shuffle=True, seed=1337)

    # Report to CodeCarbon and Weights & Biases
    training_args.report_to = ["codecarbon","wandb"]

    trainer = DPOTrainer(
        model,
        ref_model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()

    if training_args.eval_strategy != "no":
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

# Full training
# accelerate launch --config_file=dpo-accelerate-config.yml --num_processes 8 \
# dpo-tucano.py \
#--dataset_name "nicholasKluge/reward-aira-dataset" \
#--model_name_or_path "TucanoBR/Tucano-1b1-Instruct" \
#--do_train True \
#--do_eval True \
#--logging_steps 100 \
#--eval_strategy steps \
#--eval_steps 200 \
#--output_dir Tucano-1b1-DPO \
#--learning_rate 1e-6 \
#--weight_decay 0.1 \
#--lr_scheduler_type "cosine" \
#--beta 0.5 \
#--label_smoothing 0.0 \
#--loss_type "sigmoid" \
#--dataset_num_proc 8 \
#--num_train_epochs 3 \
#--per_device_train_batch_size 4 \
#--per_device_eval_batch_size 4 \
#--gradient_accumulation_steps 1 \
#--save_strategy "steps" \
#--save_steps 1000 \
#--seed 1337 \
#--data_seed 1337 \
#--bf16 True \
#--tf32 True