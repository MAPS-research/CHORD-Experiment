#!/bin/bash
#SBATCH --job-name=r10-gen-elfL
#SBATCH --output=logs/gen10/gen_elfL_%j.out
#SBATCH --error=logs/gen10/gen_elfL_%j.err
#SBATCH --time=12:00:00
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
# Repeated Table-2 protocol: ELF-L at the paper's operating point (64-step SDE,
# self-cond CFG 4, experiments/generator_eval/configs/elf_l_sampling_steps64_cfg4.yaml), 10 seeds x 500 samples,
# sampled by the official ELF code, then imported with normalize_samples.
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} TOKENIZERS_PARALLELISM=false
CHORD=${CHORD_ROOT}
OUT=$CHORD/outputs/casestudy/unconditional_generation/samples; mkdir -p $OUT
nvidia-smi --query-gpu=name,memory.total --format=csv
rc=0
for k in 01 02 03 04 05 06 07 08 09 10; do
  SEED=$((20260800 + 10#$k))
  F=$OUT/r10-elfL-s$k.jsonl
  if [ -f $F ] && [ $(wc -l < $F) -ge 500 ]; then echo "skip $F"; continue; fi
  conda activate elf
  cd ${ELF_ROOT:-third_party/ELF}
  RAW=outputs/chord_unconditional_generation/seed_$SEED; rm -rf $RAW
  NGPU=1 bash scripts/launch.sh eval src/configs/training_configs/train_owt_ELF-L.yml \
      --checkpoint_path embedded-language-flows/ELF-L-owt-torch --seed $SEED \
      --config_override use_bf16=true --config_override use_compile=true --config_override use_wandb=false \
      --config_override global_batch_size=none --config_override batch_size=50 \
      --config_override num_samples=500 --config_override output_dir=$RAW \
      --config_override sampling_configs_path=$CHORD/experiments/generator_eval/configs/elf_l_sampling_steps64_cfg4.yaml || { rc=1; continue; }
  GEN=$(ls $RAW/*/all_generated_*.jsonl | head -1)
  source "${CONDA_ACTIVATE:-conda}" activate chord-gpu 2>/dev/null || conda activate chord-gpu
  cd $CHORD; export PYTHONPATH=$PWD
  python -m chord.data.generators.normalize_samples --format jsonl --input-key generated --model ELF-L-OWT \
      --run-id r10-elfL-s$k --seed $SEED --input ${ELF_ROOT:-third_party/ELF}/$GEN --output $F.tmp && mv $F.tmp $F || rc=1
done
echo "GEN_EXIT=$rc"; exit $rc
