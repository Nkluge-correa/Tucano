#!/bin/bash -l
#SBATCH --account=the_tucanos
#SBATCH --partition=mlgpu_short
#SBATCH --job-name=Tucano-eval-pt-harness
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

workdir=/lustre/mlnvme/data/the_tucanos/Tucano/evaluations/lm-pt-evaluation
mkdir -p $workdir
cd $workdir

huggingface-cli login --token None

out=/lustre/mlnvme/data/the_tucanos/run_outputs/out.$SLURM_JOB_ID
err=/lustre/mlnvme/data/the_tucanos/run_outputs/err.$SLURM_JOB_ID

echo "# [$SLURM_JOB_ID] `date` " > ${out}

srun --cpu-bind=none python3 /lustre/mlnvme/data/the_tucanos/Tucano/evaluations/lm-pt-evaluation/lm_eval \
--model huggingface \
--model_args pretrained="TucanoBR/Tucano-160m" \
--tasks "assin2_rte,assin2_sts,bluex,enem_challenge,faquad_nli,hatebr_offensive,oab_exams,portuguese_hate_speech,tweetsentbr" \
--batch_size "auto" \
--num_fewshot 15,10,3,3,15,25,3,25,25 \
--device "cuda:0" \
--output_path "path/to/output" 1>$out 2>$err

echo "# [$SLURM_JOB_ID] `date` " >> ${out}

