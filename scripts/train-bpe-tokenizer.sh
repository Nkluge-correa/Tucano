#!/bin/bash -l
#SBATCH --account=the_tucanos
#SBATCH --partition=vlm_long
#SBATCH --job-name=bpe-tokenizer-training
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

srun --cpu-bind=none python3 /lustre/mlnvme/data/the_tucanos/Tucano/train-bpe-tokenizer.py \
--train_dataset_dir "/lustre/mlnvme/data/the_tucanos/gigaverbo-v7-tokenized" \
--text_column "text" \
--seed 1337 \
--num_samples 2000000 \
--batch_size 10000 \
--vocab_size 32000 \
--output_dir "/lustre/mlnvme/data/the_tucanos/Tucano/new-tokenizer" \
--hub_tokenizer_name "username/new-BPE-tokenizer" \
--token None \
--train_new_from_old True \
--cache_dir "/lustre/mlnvme/data/the_tucanos/cache" \
--max_length 2048 1>$out 2>$err

chmod -R g+rw /lustre/mlnvme/data/the_tucanos/cache/*

echo "# [$SLURM_JOB_ID] `date` " >> ${out}

