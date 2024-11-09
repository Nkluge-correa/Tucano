#!/bin/bash -l
#SBATCH --account=the_tucanos
#SBATCH --partition=sgpu_long
#SBATCH --job-name=Tucano-training
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=32
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:a100:4
#SBATCH --exclusive

source /lustre/mlnvme/data/the_tucanos/modules_amd.sh
source /lustre/mlnvme/data/the_tucanos/venv_amd/bin/activate
which python3

ulimit -c 0

echo `date`

export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=32
export HF_DATASETS_CACHE="/lustre/mlnvme/data/the_tucanos/cache"

MASTER_ADDR="$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)"
export MASTER_ADDR="$(nslookup "$MASTER_ADDR" | grep -oP '(?<=Address: ).*')"
export MASTER_PORT=12340
export WORLD_SIZE=16

workdir=/lustre/mlnvme/data/the_tucanos
mkdir -p $workdir
cd $workdir

out=$workdir/run_outputs/out.$SLURM_JOB_ID
err=$workdir/run_outputs/err.$SLURM_JOB_ID

echo "# [$SLURM_JOB_ID] `date` " > ${out}

srun --cpu-bind=none python3 /lustre/mlnvme/data/the_tucanos/Tucano/train-tucano.py \
 --specs "/lustre/mlnvme/data/the_tucanos/Tucano/specs-tucano-2b4.yml" \
 --slurm-job-id $SLURM_JOB_ID --hardware "a100" 1>$out 2>$err

echo "# [$SLURM_JOB_ID] `date` " >> ${out}