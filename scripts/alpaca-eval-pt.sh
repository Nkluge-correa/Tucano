#!/bin/bash -l
#SBATCH --account=the_tucanos
#SBATCH --partition=mlgpu_long
#SBATCH --job-name=Alpaca-Eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=16
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --exclusive

source /lustre/mlnvme/data/the_tucanos/modules_amd.sh
source /lustre/mlnvme/data/the_tucanos/venv_amd/bin/activate
which python3

ulimit -c 0

echo `date`

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=16
export HF_DATASETS_CACHE="/lustre/mlnvme/data/the_tucanos/cache"
export HUGGINGFACE_HUB_CACHE="/lustre/mlnvme/data/the_tucanos/cache"

workdir=/lustre/mlnvme/data/the_tucanos
mkdir -p $workdir
cd $workdir

out=$workdir/run_outputs/out.$SLURM_JOB_ID
err=$workdir/run_outputs/err.$SLURM_JOB_ID

huggingface-cli login --token "your_token_here"

echo "# [$SLURM_JOB_ID] `date` " > ${out}

srun --cpu-bind=none python3 /lustre/mlnvme/data/the_tucanos/alpaca-eval-pt.py \
--model_id TucanoBR/Tucano-2b4-Instruct \
--attn_implementation flash_attention_2 \
--precision bfloat16 \
--max_new_tokens 2048 \
--do_sample True \
--repetition_penalty 1.2 \
--temperature 0.1 \
--top_k 10 \
--top_p 0.1 \
--output_folder path/to/folder 1>$out 2>$err

echo "# [$SLURM_JOB_ID] `date` " >> ${out}
