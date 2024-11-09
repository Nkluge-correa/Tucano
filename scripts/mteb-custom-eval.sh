#!/bin/bash -l
#SBATCH --account=the_tucanos
#SBATCH --partition=mlgpu_long
#SBATCH --job-name=mntp-simcse-eval
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
pip3 install llm2vec -q
pip3 install llm2vec[evaluation] -q

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

echo "# [$SLURM_JOB_ID] `date` " > ${out}

huggingface-cli login --token "your_token"
wandb login "your_token"

srun --cpu-bind=none python3 /lustre/mlnvme/data/the_tucanos/mteb-custom-eval.py \
--base_model_name_or_path "TucanoBr/Tucano-1b1-mntp" \
--peft_model_name_or_path "TucanoBR/Tucano-1b1-mntp-simcse" \
--output_dir "Tucano-1b1-mntp-simcse" \
--is_custom 1>$out 2>$err

echo "# [$SLURM_JOB_ID] `date` " >> ${out}

