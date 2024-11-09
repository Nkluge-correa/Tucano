#!/bin/bash -l
#SBATCH --account=the_tucanos
#SBATCH --partition=vlm_long
#SBATCH --job-name=dedup-gigaverbo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=7-00:00:00
#SBATCH --mem=1900G
#SBATCH --exclusive

source /lustre/mlnvme/data/the_tucanos/modules_intel.sh
source /lustre/mlnvme/data/the_tucanos/venv_intel/bin/activate
which python3
###pip3 install text-dedup -q

ulimit -c 0

echo `date`

export OMP_NUM_THREADS=64
export HF_DATASETS_CACHE="/lustre/mlnvme/data/the_tucanos/cache"

workdir=/lustre/mlnvme/data/the_tucanos
mkdir -p $workdir
cd $workdir

out=$workdir/run_outputs/out.$SLURM_JOB_ID
err=$workdir/run_outputs/err.$SLURM_JOB_ID

huggingface-cli login --token your_token_here

echo "# [$SLURM_JOB_ID] `date` " > ${out}

srun --cpu-bind=none python -m text_dedup.exact_hash \
  --path "TucanoBR/GigaVerbo" \
  --name "default" \
  --split "train" \
  --cache_dir "/lustre/mlnvme/data/the_tucanos/cache" \
  --output "/lustre/mlnvme/data/the_tucanos/GigaVerbo-dedup" \
  --column "text" \
  --batch_size 10000 \
  --use_auth_token True 1>$out 2>$err

chmod -R g+rw /lustre/mlnvme/data/the_tucanos/cache/*

echo "# [$SLURM_JOB_ID] `date` " >> ${out}