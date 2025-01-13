#!/bin/bash -l
#SBATCH --account=the_tucanos
#SBATCH --partition=lm_long
#SBATCH --job-name=tokenize-sft
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=7-00:00:00
#SBATCH --mem=1900G
#SBATCH --exclusive

source /lustre/mlnvme/data/the_tucanos/modules_intel.sh
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

srun --cpu-bind=none python3 /lustre/mlnvme/data/the_tucanos/tokenize-sft.py \
--token "your_token_here" \
--cache_dir "/lustre/mlnvme/data/the_tucanos/cache" \
--num_proc 64 \
--system_prompt "Um bate-papo entre um usuário curioso e um assistente de inteligência artificial. O nome do assistente é Tucano, um grande modelo de linguagem desenvolvido para a geração de texto em português. O assistente dá respostas úteis, detalhadas e educadas em português às perguntas do usuário." \
--tokenizer_name "TucanoBR/Tucano-2b4" \
--chat_template "{% for message in messages %}{% if message['role'] == 'user' %}{{ '\\n Usuário: ' + message['content'] }}{% elif message['role'] == 'system' %}{{ message['content'] }}{% elif message['role'] == 'assistant' %}{{ '\\n Assistente: '  + message['content'] + eos_token }}{% endif %}{% endfor %}" \
--ignore_index -100 \
--threshold 0.8 \
--test_size 1000 \
--seed 1337 \
--output_dir "/lustre/mlnvme/data/the_tucanos/tucano-sft-4096" \
--n_chunks 5 >$out 2>$err

chmod -R g+rw /lustre/mlnvme/data/the_tucanos/cache/*

echo "# [$SLURM_JOB_ID] `date` " >> ${out}

