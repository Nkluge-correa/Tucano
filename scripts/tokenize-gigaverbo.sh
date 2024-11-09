#!/bin/bash -l
#SBATCH --account=the_tucanos
#SBATCH --partition=vlm_long
#SBATCH --job-name=tokenize-gigaverbo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=7-00:00:00
#SBATCH --mem=1900G
#SBATCH --exclusive

source /lustre/mlnvme/data/the_tucanos/modules_intel.sh
source /lustre/mlnvme/data/the_tucanos/venv_intel/bin/activate
which python3

ulimit -c 0

echo `date`

export OMP_NUM_THREADS=64
export HF_DATASETS_CACHE="/lustre/mlnvme/data/the_tucanos/cache"

workdir=/lustre/mlnvme/data/the_tucanos
mkdir -p $workdir
cd $workdir

out=$workdir/run_outputs/out.$SLURM_JOB_ID
err=$workdir/run_outputs/err.$SLURM_JOB_ID

echo "# [$SLURM_JOB_ID] `date` " > ${out}

srun --cpu-bind=none python3 /lustre/mlnvme/data/the_tucanos/Tucano/gigaverbo/tokenize-gigaverbo.py \
--output-dir "/lustre/mlnvme/data/the_tucanos/gigaverbo-tokenized" \
--cache-dir "/lustre/mlnvme/data/the_tucanos/cache" \
--datasets-dir "/lustre/mlnvme/data/the_tucanos/gigaverbo-not-tokenized" \
--tokenizer-name "TucanoBR/Tucano-1b1" \
--block-size 2048 \
--token None \
--num-proc 64 \
--seed 1337 \
--n-chunks 30 1>$out 2>$err

chmod -R g+rw /lustre/mlnvme/data/the_tucanos/cache/*

echo "# [$SLURM_JOB_ID] `date` " >> ${out}
