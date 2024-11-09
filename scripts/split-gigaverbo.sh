#!/bin/bash -l
#SBATCH --account=the_tucanos
#SBATCH --partition=vlm_long
#SBATCH --job-name=split-gigaverbo
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

srun --cpu-bind=none python3 /lustre/mlnvme/data/the_tucanos/Tucano/gigaverbo/split-gigaverbo.py \
--dataset_name "TucanoBR/GigaVerbo" \
--dataset_split "train" \
--token None \
--cache_dir "/lustre/mlnvme/data/the_tucanos/cache" \
--num_proc 64 \
--output_dir "/lustre/mlnvme/data/the_tucanos/gigaverbo-not-tokenized" \
--confidence_threshold 0.98 1>$out 2>$err

echo "# [$SLURM_JOB_ID] `date` " >> ${out}

