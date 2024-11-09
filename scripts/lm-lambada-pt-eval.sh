#!/bin/bash -l
#SBATCH --account=the_tucanos
#SBATCH --partition=mlgpu_short
#SBATCH --job-name=Tucano-eval-lambada-pt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --exclusive

source /lustre/mlnvme/data/the_tucanos/modules_amd.sh
source /lustre/mlnvme/data/the_tucanos/venv_amd_eval/bin/activate
which python3

ulimit -c 0

echo `date`

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=16
export HF_DATASETS_CACHE="/lustre/mlnvme/data/the_tucanos/cache"

workdir=/lustre/mlnvme/data/the_tucanos
mkdir -p $workdir
cd $workdir

out=$workdir/run_outputs/out.$SLURM_JOB_ID
err=$workdir/run_outputs/err.$SLURM_JOB_ID

echo "# [$SLURM_JOB_ID] `date` " > ${out}

srun --cpu-bind=none python3 /lustre/mlnvme/data/the_tucanos/Tucano/evaluations/lm-lambada-pt-eval.py \
--model_name "TucanoBR/Tucano-160m" \
--revision "main" \
--token None \
--max_new_tokens 10 \
--tf32 True \
--bf16 False \
--output_path "path/to/output" 1>$out 2>$err

echo "# [$SLURM_JOB_ID] `date` " >> ${out}
