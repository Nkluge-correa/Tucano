import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.distributed as dist

from transformers import default_data_collator, AutoTokenizer, AutoConfig, AutoModelForCausalLM
from specifications import TrainingArguments
from llama_modeling import LlamaForCausalLM

from liger_kernel.transformers import apply_liger_kernel_to_llama

from codecarbon import EmissionsTracker
import numpy as np
import datasets
import argparse
import logging
import wandb
import yaml
import time
import math
import sys
import os


def main(specs, slurm_job_id, hardware):

    # import torch

    if "WORLD_SIZE" and "SLURM_PROCID" in os.environ:

        world_size = int(os.environ["WORLD_SIZE"])

        if world_size > 1:

            rank = int(os.environ['SLURM_PROCID']) # SLURM_PROCID is the rank of the current process in SLURM

            dist.init_process_group(
                backend="nccl",
                world_size=world_size, 
                rank=rank
            )

            # Set the device to the current rank.
            device = f"cuda:{rank % torch.cuda.device_count()}"
            torch.cuda.set_device(device)
            master_process = rank == 0 # The first process is the master process.
            ddp = True
            print(f"Running DDP via '{dist.get_backend()}' backend. Process rank: {dist.get_rank()}. World size: {dist.get_world_size()}.")
        
        else:
            # If the world size is 1, then we are not using distributed training.
            # Set the device to the current rank.
            rank = 0
            device = "cuda:0"
            torch.cuda.set_device(device)
            master_process = True
            ddp = False
            print("Running single process training.")

    else:
        raise ValueError("WORLD_SIZE or SLURM_PROCID environment variable is not set. This script is intended to be run with SLURM.")

    # Load the training arguments from the specifications
    with open(specs, "r") as stream:
        kwargs = yaml.safe_load(stream)
    args = TrainingArguments(**kwargs)

    # If we are resuming from a checkpoint, we need to get the slurm job id from the checkpoint directory,
    # and use it as the indentification for the current job.
    if args.resume_from_checkpoint:
        slurm_job_id = args.resume_from_checkpoint.split("slurm_job_")[-1].split("/")[0]

    # Add the slurm job id to the `checkpoint_dir`
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, f"slurm_job_{slurm_job_id}")

    # Create the checkpoint directory if it does not exist.
    if master_process:
        
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        # Create a log file to store the training logs
        log_file = os.path.join(args.checkpoint_dir, f"logs-{slurm_job_id}.txt")
        with open(log_file, "a") as f: 
            pass

    # Set the random state seed for reproducibility
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set the precision for matrix multiplication, the TF32 mode and the CuDNN TF32 mode, 
    # and the model precision to bfloat16 if `bf16` is set to True. 
    torch.set_float32_matmul_precision(args.mat_mul_precision)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = args.bf16
    precision = torch.bfloat16 if args.bf16 else torch.float32

    # Nothing fancy here, just a simple logger.
    logger = logging.getLogger(f"Tucano-{slurm_job_id}")

    logging.basicConfig(
        format="%(name)s - %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Load the tokenizer from HuggingFace. 
    if args.tokenizer_name is not None:

        # `AutoTokenizer`:https://huggingface.co/docs/transformers/v4.40.1/en/model_doc/auto#transformers.AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, 
            **{
            "cache_dir": args.cache_dir,
            "use_fast": True, # Always use the fast tokenizer
            "token": args.hub_token,
            }
        )
    
    # Potentially load the model from a previous checkpoint.
    if args.resume_from_checkpoint:
        
        checkpoint_path = os.path.join(args.resume_from_checkpoint)
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path, torch_dtype=precision, attn_implementation="flash_attention_2", cache_dir=args.cache_dir)
        
        # Adding this line automatically monkey-patches the model with the optimized Liger kernels, i.e.,
        # a collection of Triton kernels designed specifically for LLM training ([source](https://github.com/linkedin/Liger-Kernel))
        apply_liger_kernel_to_llama() 

        if master_process:
            logger.info(f"Resumed model from checkpoint: {checkpoint_path}")
            with open(log_file, "a") as f:
                f.write(f"Resumed model from checkpoint: {checkpoint_path}\n")
    
    else:

        if master_process:
            logger.info(f"Initializing model from mlnvme.")
            with open(log_file, "a") as f:
                f.write(f"Initializing model from mlnvme.\n")

        # Create a model from the AutoConfig
        configuration = AutoConfig.from_pretrained(
            "nicholasKluge/TeenyTinyLlama-160m", # basic Llama model configuration 
            **{
            "cache_dir": args.cache_dir,
            "token": args.hub_token,
            "output_hidden_states": args.output_hidden_states,
            "hidden_size": args.hidden_size,
            "intermediate_size": args.hidden_size * 4 if args.intermediate_size is None else args.intermediate_size,
            "max_position_embeddings": args.max_position_embeddings,
            "num_attention_heads": args.num_attention_heads,
            "num_hidden_layers": args.num_hidden_layers,
            "num_key_value_heads": args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "unk_token_id": tokenizer.unk_token_id,
            "torch_dtype": precision,
            "vocab_size": len(tokenizer),
            "use_cache": args.use_cache,
            "hidden_act": args.hidden_act,
            }
        )

        # Create an instance of the model using the configuration.
        model = AutoModelForCausalLM.from_config(configuration, attn_implementation="flash_attention_2", **{"torch_dtype": precision})

        # Change the model's `name_or_path`.  
        model.config.name_or_path = args.hub_model_id
        
        # Adding this line automatically monkey-patches the model with the optimized Liger kernels, i.e.,
        # a collection of Triton kernels designed specifically for LLM training ([source](https://github.com/linkedin/Liger-Kernel))
        apply_liger_kernel_to_llama() 
        
        # Print the number of trainable parameters in the model.
        if master_process:
            params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Number of trainable parameters: {params:,}")
            with open(log_file, "a") as f:
                f.write(f"Number of trainable parameters: {params:,}\n")

    # Set the gradient checkpointing for the model
    if args.gradient_checkpointing:

        if master_process:
            logger.info(f"Gradient checkpointing enabled.")

        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    model.to(device)

    if ddp:
        # Initialize the DistributedDataParallel model.
        #`DistributedDataParallel`: https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
        model = DDP(
            model, 
            device_ids=[rank % torch.cuda.device_count()],
            static_graph=True,
        ) 

    raw_model = model.module if ddp else model # Unwrap version of the model if it is wrapped in DDP.

    # Load the dataset and create the dataloaders
    if args.sanity_check:

        # Create a list of tokens to be used for the sanity check
        input_ids = [list(range(args.max_position_embeddings))]  * 100000
        
        dataset = datasets.Dataset.from_dict(
            {
                "input_ids": input_ids,
            }
        ).with_format("torch")

        dataset = dataset.train_test_split(test_size=0.1, seed=args.seed)
        train_dataset, val_dataset = dataset["train"], dataset["test"]
    
    else:

        # List all jsonl files in the train and validation dataset directories
        train_dataset_files = sorted([os.path.join(args.train_dataset_dir, f) for f in os.listdir(args.train_dataset_dir) if f.endswith(".jsonl")], key=lambda x: int(x.split("_")[-1].split(".")[0]))
        val_dataset_files = [os.path.join(args.val_dataset_dir, f) for f in os.listdir(args.val_dataset_dir) if f.endswith(".jsonl")]

        # Load the datasets from disk
        train_dataset = datasets.load_dataset(
            "json",
            data_files=train_dataset_files,
            split='train',
            num_proc=len(train_dataset_files),
            cache_dir=args.cache_dir,
        )

        val_dataset = datasets.load_dataset(
            "json",
            data_files=val_dataset_files,
            split='train',
            num_proc=len(val_dataset_files),
            cache_dir=args.cache_dir,
        )

        train_dataset = train_dataset.with_format("torch")
        val_dataset = val_dataset.with_format("torch")
        

    # Initialize the DistributedSampler:
    # `DistributedSampler`: https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler	
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
        )
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
        )
    
    # Create the dataloaders.
    # `DataLoader`: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    dataloader_generator = torch.Generator()
    dataloader_generator.manual_seed(args.seed)
    
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        collate_fn=default_data_collator,  # `default_data_collator` function provided by the `transformers` library : https://huggingface.co/docs/transformers/main_classes/data_collator#transformers.default_data_collator
        batch_size=args.micro_batch_size,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers_for_dataloader,
        generator=dataloader_generator,
        prefetch_factor=4, # The number of samples loaded in advance by the dataloader.
    )

    validation_dataloader = DataLoader(
        val_dataset,
        sampler=val_sampler,
        collate_fn=default_data_collator,
        batch_size=args.eval_micro_batch_size,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers_for_dataloader,
        prefetch_factor=4,
        # We do not need to shuffle the validation set, so we dont pass in the generator
    )

    # Calculate the gradient accumulation steps
    assert args.total_batch_size % (args.micro_batch_size * args.max_position_embeddings * world_size) == 0, "Make sure your `total_batch_size` is divisible by `micro_batch_size` * `max_position_embeddings` * `world_size`"
    # The number of gradient accumulation steps is estimated based on how many tokens you want (`total_batch_size`) 
    # to be processed in a single step. 
    gradient_accumulation_steps = args.total_batch_size // (args.micro_batch_size * args.max_position_embeddings * world_size)

    # Now we need to calculate the math around the number of steps per epoch and the total number of steps.
    # Initially, we will set the number of steps per epoch to the number of batches in the training dataloader.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)

    # The number of learning rate decay iterations is set to be a percentage of the total number of steps.
    # e.g., 0.9 means we will decay the learning rate over 90% of the total number of steps.
    lr_decay_iters = (max_steps - args.warmup_steps) * args.lr_decay_iters_coef

    # Our scheduler will start with a warmup phase, where the learning rate will increase linearly from 0 to the initial learning rate
    # over the first n `warmup_steps` of the training steps. Then, the learning rate will decrease following the cosine function until 
    # it reaches the minimum learning rate. After that, the learning rate will remain constant.
    def lr_scheduler(it):
        # 1) Linear warmup for `warmup_steps` steps
        if it < args.warmup_steps:
            return args.max_learning_rate * it / args.warmup_steps
        # 2) If it > `lr_decay_iters``, return min learning rate
        if it > lr_decay_iters:
            return args.min_learning_rate
        # 3) In between, use cosine decay down to `min_learning_rate`
        decay_ratio = (it - args.warmup_steps) / (lr_decay_iters - args.warmup_steps)
        assert 0 <= decay_ratio <= 1
        
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return args.min_learning_rate + coeff * (args.max_learning_rate - args.min_learning_rate)

    # Now, we create our Optimizer. First, we will split weights in two groups, 
    # one with weight decay and the other not.
    # These strings `["bias", "layer_norm.weight"]` represent parameter names that should not be subject to weight decay during optimization. 
    # Weight decay is a regularization technique used during training to prevent overfitting by penalizing large weights.
    no_decay = ["bias", "layer_norm.weight"]

    # The first dictionary corresponds to parameters with weight decay (regularization) applied to them (non-bias and non-layer_norm.weight parameters).
    # The second dictionary corresponds to parameters without weight decay (regularization) applied to them (bias and layer_norm.weight parameters).
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    # `AdamW`: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, 
        lr=args.max_learning_rate,
        eps=args.eps,
        betas=(args.beta1, args.beta2),
        fused=True if device_type=="cuda" else False
    )

    # Resume the optimizer from the checkpoint
    if args.resume_from_checkpoint:

        if master_process:
            logger.info(f"Resumed optimizer from checkpoint: {args.resume_from_checkpoint}")
            with open(log_file, "a") as f:
                f.write(f"Resumed optimizer from checkpoint: {args.resume_from_checkpoint}\n")

        checkpoint = os.path.join(args.resume_from_checkpoint, 'checkpoint.pt')
        checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'), weights_only=False)
        optimizer.load_state_dict(checkpoint['optimizer'])

    if master_process:

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Length of the train dataLoader = {len(train_dataloader)}.")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.micro_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
        logger.info(f"  Total batch size (w. parallel, distributed & accumulation) = {args.micro_batch_size * gradient_accumulation_steps * world_size}")
        logger.info(f"  Total batch size (in tokens) = {args.total_batch_size}")
        logger.info(f"  Total optimization steps = {max_steps}")

        if args.resume_from_checkpoint is None:
            with open(log_file, "a") as f:
                f.write("***** Resume training *****\n")
                f.write(f"  Num examples = {len(train_dataset)}\n")
                f.write(f"  Length of the train dataLoader = {len(train_dataloader)}.\n")
                f.write(f"  Num Epochs = {args.num_train_epochs}\n")
                f.write(f"  Instantaneous batch size per device = {args.micro_batch_size}\n")
                f.write(f"  Gradient Accumulation steps = {gradient_accumulation_steps}\n")
                f.write(f"  Total batch size (w. parallel, distributed & accumulation) = {args.micro_batch_size * gradient_accumulation_steps * world_size}\n")
                f.write(f"  Total batch size (in tokens) = {args.total_batch_size}\n")
                f.write(f"  Total optimization steps = {max_steps}\n")

    # If we are resuming from a checkpoint, we need to get the current step and the number of completed steps.
    # completed_steps are the optimization steps that have been completed so far.
    if args.resume_from_checkpoint:
        # Get the current step and the iteration count from the checkpoint.
        resume_step = int(checkpoint['resume_step'])
        iter_count = int(checkpoint['iteration'])
        epoch = int(checkpoint['epoch'])

        if epoch > 1:
            train_sampler.set_epoch(epoch)

        if master_process:
            logger.info(f"Resuming training from step {resume_step}.")
            with open(log_file, "a") as f:
                f.write(f"Resuming training from step {resume_step}.\n")
    
    else:
        resume_step = 0
        iter_count = 0
        epoch = 1

    # Initialize  W&B and CodeCarbon
    if master_process:

        if args.wandb_token is not None: 

            # Login to wandb.
            wandb.login(key=args.wandb_token)

            # Initialize wandb.
            wandb.init(
                project="Tucano", 
                notes="Training a Llama model on a custom Portuguese dataset.",
                tags=["Language Modeling", "Portuguese"],
                name=f"""{args.wandb_id}-{time.strftime("%d-%m-%Y")}""",
                config=kwargs,
                resume="allow",
                id=args.wandb_id,
            )
        
        # We would also like to track the energy consumption of the training process. We are going to use the `codecarbon` library
        # to do this. We need to initialize the `EmissionsTracker` and then track the energy consumption on [only the main process](https://github.com/mlco2/codecarbon/issues/544).
        # `EmissionsTracker`: https://mlco2.github.io/codecarbon/usage.html#explicit-object
        tracker = EmissionsTracker(
            project_name="Tucano",
            log_level="critical", # Set to "critical" to silence codecarbon.
            output_dir=args.checkpoint_dir,
            output_file=f"emissions_{slurm_job_id}.csv",
            tracking_mode='machine', # We are tracking the energy consumption of the whole machine.
        )

        logger.info(f'Geo Location: ISO: {tracker._geo.country_iso_code} | Country: {tracker._geo.country_name} | Region : {tracker._geo.region}')
        tracker.start()

    # MFU calculation parameters
    C = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if hardware == "a100":
        P = 300e12 if args.bf16 else 150e12
    elif hardware == "a40":
        P = 150e12 if args.bf16 else 37e12
    else:
        raise ValueError("Hardware not supported.")

    # Set the model to training mode.
    model.train()

    # Create an iterator from the train dataloader
    iter_train_dataloader = iter(train_dataloader)

    if master_process:
        logger.info(f"Epoch {epoch} of {math.ceil(args.num_train_epochs)}")
        with open(log_file, "a") as f:
            f.write(f"Epoch {epoch} of {math.ceil(args.num_train_epochs)}\n")

    # Start the training loop
    for completed_steps in range(max_steps):

        # Skip the steps that have already been completed when resuming from a checkpoint.
        if resume_step > completed_steps:
            for micro_step in range(gradient_accumulation_steps):
                try:
                    next(iter_train_dataloader)
                except StopIteration:
                    iter_train_dataloader = iter(train_dataloader)
                    next(iter_train_dataloader)
            continue
            
        # Reset the sampler if we started looping over the dataloader
        if iter_count > len(train_dataloader) - 1:
            if epoch < math.ceil(args.num_train_epochs):
                epoch += 1
                iter_count = 0
                train_sampler.set_epoch(epoch)
                if master_process:
                    logger.info(f"Epoch {epoch} of {math.ceil(args.num_train_epochs)}")
                    with open(log_file, "a") as f:
                        f.write(f"Epoch {epoch} of {math.ceil(args.num_train_epochs)}\n")
         
        # Evaluate the model on the validation set at every checkpointing step
        if completed_steps % args.checkpointing_steps == 0 and completed_steps > 0:
            
            if master_process:
                logger.info("***** Running validation *****")
            
            model.eval()

            with torch.no_grad():

                val_loss_accum = 0.0

                for _, batch in enumerate(validation_dataloader):

                    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                    
                    with torch.autocast(device_type=device_type, dtype=precision):
                        loss = model(
                            input_ids=batch["input_ids"],
                            labels=batch["input_ids"],
                        ).loss

                    loss = loss / len(validation_dataloader)
                    val_loss_accum += loss.detach()

            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            
            model.train()

            if master_process:

                logger.info(f"Validation | step: {completed_steps:5d} | loss: {val_loss_accum.item():.4f} | kWh: {tracker._total_energy.kWh:.2f}")

                with open(log_file, "a") as f:
                    f.write(f"Validation | step: {completed_steps:5d} | loss: {val_loss_accum.item():.4f} |  kWh: {tracker._total_energy.kWh:.2f}\n")

                if args.wandb_token is not None:
                    wandb.log({"val_loss": val_loss_accum.item()})

                # Save the model checkpoint.
                output_dir = os.path.join(args.checkpoint_dir, f"step_{completed_steps:05d}")
                os.makedirs(output_dir, exist_ok=True)
                checkpoint_path = os.path.join(output_dir, "checkpoint.pt")
                raw_model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)

                torch.save(
                    {
                    'resume_step' : completed_steps + 1,
                    'iteration': iter_count,
                    'epoch': epoch,
                    'config': raw_model.config,
                    'optimizer': optimizer.state_dict(),
                    }, 
                    checkpoint_path,
                )
            
                # Convert the checkpoint to a HuggingFace and push it to the hub.
                if args.push_to_hub and args.hub_token is not None and args.hub_model_id is not None:
                    
                    hub_model_id = args.hub_model_id + f"-{completed_steps}"
                    raw_model.push_to_hub(hub_model_id, token=args.hub_token, private=True)
                    if args.tokenizer_name is not None:
                        tokenizer.push_to_hub(hub_model_id, token=args.hub_token)

                # Flush the codecarbon tracker at the end of the validation step.
                tracker.flush()
            
            # Set barrier to ensure that all processes have finished the validation step before continuing.
            if ddp:
                dist.barrier()

        # We are timing the training loop to measure the MFU.
        t0 = time.time()

        # Initiate a counter for the accumulated loss
        accumulated_loss = 0.0
        
        # Perform one optimization step
        optimizer.zero_grad(set_to_none=True)
        
        # Perform the gradient accumulation inner loop
        # Intersting points to note: https://discuss.pytorch.org/t/does-number-of-gradient-accumulation-steps-affect-models-performance/85859/5
        for micro_step in range(gradient_accumulation_steps):

            # Get the next batch
            try:
                batch = next(iter_train_dataloader)
            except StopIteration:
                iter_train_dataloader = iter(train_dataloader)
                batch = next(iter_train_dataloader)

            # Move the batch to the device
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            
            # Autocast is a PyTorch context manager that enables mixed precision training.
            # `torch.autocast`: https://pytorch.org/docs/stable/amp.html#torch.autocast
            with torch.autocast(device_type=device_type, dtype=precision):
                loss = model(
                            input_ids=batch["input_ids"],
                            labels=batch["input_ids"],
                        ).loss
            
            # We need to divide the loss by the gradient accumulation steps to ensure that 
            # the gradients are averaged across the steps.
            loss = loss / gradient_accumulation_steps
            accumulated_loss += loss.detach()

            # Perform the backward pass. 
            # In the case of DDP, we only need to perform to sync the gradients at the end of the gradient accumulation steps.
            if ddp:
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            loss.backward()
        
        if ddp:
            # Before we perform the gradient update, we need to average the accumulated loss across 
            # all the processes in the DDP group.
            dist.all_reduce(accumulated_loss, op=dist.ReduceOp.AVG)

        # Clip the gradients to prevent the exploding gradient problem
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Determine the learning rate for the current step
        lr = lr_scheduler(completed_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Perform the optimizer step
        optimizer.step()
        torch.cuda.synchronize() # Wait for the processess to synchronize before we measure the time.
        iter_count += 1
        t1 = time.time()

        # Calculate the time taken for the step
        dt = t1 - t0
        tokens_processed = args.micro_batch_size * args.max_position_embeddings * gradient_accumulation_steps * world_size
        tokens_per_sec = tokens_processed / dt
        tokens_processed_per_device = args.micro_batch_size * args.max_position_embeddings * gradient_accumulation_steps
        tokens_per_sec_per_device = tokens_processed_per_device / dt

        if master_process:

            used_vram = torch.cuda.memory_reserved() / (1024 ** 3) # Convert to GB
            logger.info(f"Training | step: {completed_steps + 1:5d} | loss: {accumulated_loss.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f} | VRAM: {used_vram:.2f} | MFU: {((6 * C * tokens_per_sec_per_device) / P) * 100:.2f}% | kWh: {tracker._total_energy.kWh:.2f}")
            
            # Log to the txt log file
            with open(log_file, "a") as f:
                f.write(f"Training | step: {completed_steps + 1:5d} | loss: {accumulated_loss.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f} | VRAM: {used_vram:.2f} | MFU: {((6 * C * tokens_per_sec_per_device) / P) * 100:.2f}% | kWh: {tracker._total_energy.kWh:.2f}\n")
            
            if args.wandb_token is not None:
                
                wandb.log({
                            "loss": accumulated_loss.item(),
                            "lr": lr, 
                            })
    
    # Terminate the W&B tracker and the CodeCarbon tracker at the end of the training loop.
    if master_process:
        tracker.stop()
        if args.wandb_token is not None:
            wandb.finish()

        # Save the final model checkpoint.
        output_dir = os.path.join(args.checkpoint_dir, "final-checkpoint")
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_path = os.path.join(output_dir, "checkpoint.pt")
        raw_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        torch.save(
            {
            'resume_step' : completed_steps + 1,
            'iteration': iter_count,
            'epoch': epoch,
            'config': raw_model.config,
            'optimizer': optimizer.state_dict(),
            }, 
            checkpoint_path,
        )

        # Convert the checkpoint to a HuggingFace and push it to the hub.
        if args.push_to_hub and args.hub_token is not None and args.hub_model_id is not None:
            
            hub_model_id = args.hub_model_id + "-final-checkpoint"
            raw_model.push_to_hub(hub_model_id, token=args.hub_token, private=True) 
            if args.tokenizer_name is not None:
                tokenizer.push_to_hub(hub_model_id, token=args.hub_token)

    if ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--specs",
        type=str,
        required=True,
        help="The path to the specifications file.",
    )
    parser.add_argument(
        "--slurm-job-id",
        type=str,
        required=True,
        help="The SLURM job id.",
    )
    parser.add_argument(
        "--hardware",
        type=str,
        required=True,
        help="The hardware used for training.",
    )
    args = parser.parse_args()
    main(args.specs, args.slurm_job_id, args.hardware)

# How to run this script:
# python3 train-tucano.py --specs "specs.yml" --slurm-job-id $SLURM_JOB_ID --hardware "a100"