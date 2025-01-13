#!/bin/bash -l
#SBATCH --account=the_tucanos
#SBATCH --partition=mlgpu_long
#SBATCH --job-name=ViTucano-sft
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=16
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:a40:8
#SBATCH --exclusive

source /lustre/mlnvme/data/the_tucanos/modules_amd_no_ucx.sh
source /lustre/mlnvme/data/the_tucanos/venv_amd_llava/bin/activate
which python3
###cd /lustre/mlnvme/data/the_tucanos/TinyLLaVA_Factory
export CUDA_HOME=/opt/software/easybuild-AMD/software/CUDA/12.1.1
###pip3 install --upgrade pip
###pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
###pip3 install deepspeed
###pip3 install -e "."
###pip3 install wheel
###pip3 install flash-attn --no-build-isolation
###pip3 install codecarbon

ulimit -c 0

echo `date`

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=16
export HF_DATASETS_CACHE="/lustre/mlnvme/data/the_tucanos/cache"
export HUGGINGFACE_HUB_CACHE="/lustre/mlnvme/data/the_tucanos/cache"
export TRITON_CACHE_DIR="/lustre/mlnvme/data/the_tucanos/cache"

workdir=/lustre/mlnvme/data/the_tucanos
mkdir -p $workdir
cd $workdir

out=$workdir/run_outputs/out.$SLURM_JOB_ID
err=$workdir/run_outputs/err.$SLURM_JOB_ID

huggingface-cli login --token "your_token"
wandb login "your_token"

echo "# [$SLURM_JOB_ID] `date` " > ${out}

deepspeed /lustre/mlnvme/data/the_tucanos/TinyLLaVA_Factory/tinyllava/train/train.py \
--deepspeed "/lustre/mlnvme/data/the_tucanos/TinyLLaVA_Factory/scripts/zero3.json" \
--data_path "/lustre/mlnvme/data/the_tucanos/dataset/data/sft/sft-data.json" \
--image_folder "/lustre/mlnvme/data/the_tucanos/dataset/data/sft" \
--is_multimodal True \
--conv_version "llama" \
--model_name_or_path "TucanoBR/Tucano-2b4" \
--vision_tower "google/siglip-so400m-patch14-384" \
--vision_tower2 "" \
--connector_type "mlp2x_gelu" \
--mm_vision_select_layer -2 \
--image_aspect_ratio "square" \
--attn_implementation flash_attention_2 \
--training_recipe "common" \
--tune_type_llm "full" \
--tune_type_vision_tower "frozen" \
--tune_vision_tower_from_layer 0 \
--tune_type_connector "full" \
--group_by_modality_length False \
--pretrained_model_path "/lustre/mlnvme/data/the_tucanos/checkpoints/ViTucano-2b8-projector" \
--output_dir "/lustre/mlnvme/data/the_tucanos/checkpoints/ViTucano-2b8-v1" \
--num_train_epochs 4 \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 2000 \
--save_total_limit 1 \
--learning_rate 2e-5 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 True \
--bf16 True \
--model_max_length 4096 \
--gradient_checkpointing True \
--dataloader_num_workers 8 \
--lazy_preprocess True \
--report_to "wandb" \
--tokenizer_use_fast True \
--run_name "vitucano-2b8-v1" 1>$out 2>$err

echo "# [$SLURM_JOB_ID] `date` " >> ${out}
