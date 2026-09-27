#!/bin/bash
#SBATCH --job-name=r10-gen-langflow
#SBATCH --output=logs/gen10/gen_langflow_%j.out
#SBATCH --error=logs/gen10/gen_langflow_%j.err
#SBATCH --time=06:00:00
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
  conda activate chord-generators
  F=$OUT/r10-langflow-s$k.jsonl
  if [ -f $F ] && [ $(wc -l < $F) -ge 500 ]; then echo "skip $F"; continue; fi
  python -m chord.data.generators.langflow --repo third_party/LangFlow --checkpoint third_party/ckpt/langflow-owt/model.safetensors \
    --run-id r10-langflow-s$k --seed $SEED --num-samples 500 --batch-size 50 --steps 128 --length 512 --output $F.tmp && mv $F.tmp $F || rc=1
done
echo "GEN_EXIT=$rc"; exit $rc
