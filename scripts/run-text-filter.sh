#!/bin/bash -l
#SBATCH --account=the_tucanos
#SBATCH --partition=sgpu_long
#SBATCH --job-name=filtering-gigaverbo
#SBATCH --nodes=1
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

export OMP_NUM_THREADS=32
export HF_DATASETS_CACHE="path/to/cache"
export HUGGINGFACE_HUB_CACHE="path/to/cache"

workdir=/lustre/mlnvme/data/the_tucanos
mkdir -p $workdir
cd $workdir

out0=$workdir/run_parsing/out0.$SLURM_JOB_ID
err0=$workdir/run_parsing/err0.$SLURM_JOB_ID
out1=$workdir/run_parsing/out1.$SLURM_JOB_ID
err1=$workdir/run_parsing/err1.$SLURM_JOB_ID
out2=$workdir/run_parsing/out2.$SLURM_JOB_ID
err2=$workdir/run_parsing/err2.$SLURM_JOB_ID
out3=$workdir/run_parsing/out3.$SLURM_JOB_ID
err3=$workdir/run_parsing/err3.$SLURM_JOB_ID

echo "# [$SLURM_JOB_ID] `date` " > ${out0}
echo "# [$SLURM_JOB_ID] `date` " > ${out1}
echo "# [$SLURM_JOB_ID] `date` " > ${out2}
echo "# [$SLURM_JOB_ID] `date` " > ${out3}

export CUDA_VISIBLE_DEVICES=0
export UCX_NET_DEVICES=mlx5_0:1
srun -n 1 -N 1 --gpus=1 --exclusive \
python3 /lustre/mlnvme/data/the_tucanos/run-text-classifier.py \
--model_name "TucanoBR/BERTimbau-base-text-filter" \
--directory_path "path/to/dataset" \
--file_path "path/to/dataset/chunk_36.jsonl" \
--cache_dir "path/to/cache" \
--token "your_token_here" \
--batch_size 256 \
--output_folder "path/to/output" \
--num_proc 32 \
--n_chunks 1 1>$out0 2>$err0 &

export CUDA_VISIBLE_DEVICES=1
export UCX_NET_DEVICES=mlx5_1:1
srun -n 1 -N 1 --gpus=1 --exclusive \
python3 /lustre/mlnvme/data/the_tucanos/run-text-classifier.py \
--model_name "TucanoBR/BERTimbau-base-text-filter" \
--directory_path "path/to/dataset" \
--file_path "path/to/dataset/chunk_37.jsonl" \
--cache_dir "path/to/cache" \
--token "your_token_here" \
--batch_size 256 \
--output_folder "path/to/output" \
--num_proc 32 \
--n_chunks 1 1>$out1 2>$err1 &

export CUDA_VISIBLE_DEVICES=2
export UCX_NET_DEVICES=mlx5_3:1
srun -n 1 -N 1 --gpus=1 --exclusive \
python3 /lustre/mlnvme/data/the_tucanos/run-text-classifier.py \
--model_name "TucanoBR/BERTimbau-base-text-filter" \
--directory_path "path/to/dataset" \
--file_path "path/to/dataset/chunk_38.jsonl" \
--cache_dir "path/to/cache" \
--token "your_token_here" \
--batch_size 256 \
--output_folder "path/to/output" \
--num_proc 32 \
--n_chunks 1 1>$out2 2>$err2 &

export CUDA_VISIBLE_DEVICES=3
export UCX_NET_DEVICES=mlx5_2:1
srun -n 1 -N 1 --gpus=1 --exclusive \
python3 /lustre/mlnvme/data/the_tucanos/run-text-classifier.py \
--model_name "TucanoBR/BERTimbau-base-text-filter" \
--directory_path "path/to/dataset" \
--file_path "path/to/dataset/chunk_39.jsonl" \
--cache_dir "path/to/cache" \
--token "your_token_here" \
--batch_size 256 \
--output_folder "path/to/output" \
--num_proc 32 \
--n_chunks 1 1>$out3 2>$err3 &

wait

echo "# [$SLURM_JOB_ID] `date` " >> ${out0}
echo "# [$SLURM_JOB_ID] `date` " >> ${out1}
echo "# [$SLURM_JOB_ID] `date` " >> ${out2}
echo "# [$SLURM_JOB_ID] `date` " >> ${out3}
