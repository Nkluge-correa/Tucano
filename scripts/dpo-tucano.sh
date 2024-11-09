#!/bin/bash -l
#SBATCH --account=the_tucanos
#SBATCH --partition=mlgpu_long
#SBATCH --job-name=Tucano-DPO
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=16
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:a40:8
#SBATCH --exclusive

source /lustre/mlnvme/data/the_tucanos/modules_amd.sh
source /lustre/mlnvme/data/the_tucanos/venv_amd/bin/activate
which python3
###pip3 install trl

ulimit -c 0

echo `date`

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=8
export HF_DATASETS_CACHE="/lustre/mlnvme/data/the_tucanos/cache"
export HUGGINGFACE_HUB_CACHE="/lustre/mlnvme/data/the_tucanos/cache"

workdir=/lustre/mlnvme/data/the_tucanos
mkdir -p $workdir
cd $workdir

out=$workdir/run_outputs/out.$SLURM_JOB_ID
err=$workdir/run_outputs/err.$SLURM_JOB_ID

huggingface-cli login --token "your_token"
wandb login "your_token"

echo "# [$SLURM_JOB_ID] `date` " > ${out}

accelerate launch \
--config_file="dpo-accelerate-config.yml" \
--num_processes 8 \
dpo-tucano.py \
--dataset_name "nicholasKluge/reward-aira-dataset" \
--model_name_or_path "TucanoBR/Tucano-2b4-Instruct" \
--do_train True \
--do_eval True \
--logging_steps 100 \
--eval_strategy steps \
--eval_steps 200 \
--output_dir "path/to/Tucano-2b4-Instruct-DPO" \
--learning_rate 1e-6 \
--weight_decay 0.1 \
--lr_scheduler_type "cosine" \
--beta 0.5 \
--label_smoothing 0.0 \
--loss_type "sigmoid" \
--dataset_num_proc 8 \
--num_train_epochs 3 \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
--gradient_accumulation_steps 1 \
--gradient_checkpointing True \
--save_strategy "steps" \
--save_steps 1000 \
--seed 1337 \
--data_seed 1337 \
--bf16 True \
--tf32 True 1>$out 2>$err

echo "# [$SLURM_JOB_ID] `date` " >> ${out}
