#!/bin/bash
#SBATCH --job-name=layer-depth-feat
#SBATCH --output=logs/layer_depth/featurize_%j.out
#SBATCH --error=logs/layer_depth/featurize_%j.err
#SBATCH --time=6:00:00
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
# Every-second-layer features of one Table 1 backbone (appendix "Effect of
# Extraction Layer"). Finished text sets are skipped, so a requeued job resumes.
#   SIZE=27b sbatch experiments/layer_depth/featurize.slurm      # or SIZE=9b
source "${CONDA_ACTIVATE:-conda}" activate chord-gpu 2>/dev/null || conda activate chord-gpu
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} TOKENIZERS_PARALLELISM=false
cd "${CHORD_ROOT:-$(dirname "$0")/../..}"; export PYTHONPATH=$PWD
nvidia-smi --query-gpu=name,memory.total --format=csv
S=${SIZE:?set SIZE=27b or 9b}
python -m experiments.layer_depth.featurize \
  --encoder-config experiments/counterfactual_eval/configs/encoders_chord_$S.yaml \
  --corpus-dir outputs/counterfactual/meta_eval --out-dir outputs/experiments/layer_depth/qwen35_$S
