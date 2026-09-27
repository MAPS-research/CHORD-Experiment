#!/bin/bash
#SBATCH --job-name=r10-gen-gpt2large
#SBATCH --output=logs/gen10/gen_gpt2large_%j.out
#SBATCH --error=logs/gen10/gen_gpt2large_%j.err
#SBATCH --time=04:00:00
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
# Repeated Table-2 protocol: 10 seeds x 500 samples, the paper's operating point.
# Finished seed files are skipped, so a requeued job resumes.
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} TOKENIZERS_PARALLELISM=false
cd "${CHORD_ROOT:-$(dirname "$0")/../../..}"
export PYTHONPATH=$PWD
OUT=outputs/casestudy/unconditional_generation/samples; mkdir -p $OUT
nvidia-smi --query-gpu=name,memory.total --format=csv
rc=0
for k in 01 02 03 04 05 06 07 08 09 10; do
  SEED=$((20260800 + 10#$k))
  source "${CONDA_ACTIVATE:-conda}" activate chord-gpu 2>/dev/null || conda activate chord-gpu
  F=$OUT/r10-gpt2large-s$k.jsonl
  if [ -f $F ] && [ $(wc -l < $F) -ge 500 ]; then echo "skip $F"; continue; fi
  CFG=$OUT/.gen_ar_gpt2large_$k.yaml
  printf 'model: gpt2-large\nn_samples: 500\nmax_new_tokens: 512\nmin_new_tokens: 480\ntemperature: 1.0\ntop_p: 0.95\nrepetition_penalty: 1.0\nbatch_size: 50\nseed: %s\noutput_path: %s\n' $SEED $PWD/$F.tmp > $CFG
  python -m chord.data.generate_ar --config $CFG && mv $F.tmp $F || rc=1
done
echo "GEN_EXIT=$rc"; exit $rc
