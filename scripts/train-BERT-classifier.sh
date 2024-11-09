#!/bin/bash -l
#SBATCH --account=the_tucanos
#SBATCH --partition=mlgpu_medium
#SBATCH --job-name=train-BERT-classifier
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
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

MASTER_ADDR="$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)"
export MASTER_ADDR="$(nslookup "$MASTER_ADDR" | grep -oP '(?<=Address: ).*')"
export MASTER_PORT=12340
export WORLD_SIZE=1

workdir=/lustre/mlnvme/data/the_tucanos
mkdir -p $workdir
cd $workdir

out=$workdir/run_outputs/out.$SLURM_JOB_ID
err=$workdir/run_outputs/err.$SLURM_JOB_ID

echo "# [$SLURM_JOB_ID] `date` " > ${out}

srun --cpu-bind=none python3 /lustre/mlnvme/data/the_tucanos/gigaverbo/text-filter/train-BERT-classifier.py \
--dataset-path "TucanoBR/GigaVerbo-Text-Filter" \
--model-name "neuralmind/bert-base-portuguese-cased" \
--output-dir "/lustre/mlnvme/data/the_tucanos/checkpoints" \
--learning-rate 4e-5 \
--train-batch-size 32 \
--eval-batch-size 32 \
--num-train-epochs 3 \
--weight-decay 0.01 \
--hub-token None \
--hub-model-id "userName/modelName" \
--evaluate True 1>$out 2>$err

echo "# [$SLURM_JOB_ID] `date` " >> ${out}

