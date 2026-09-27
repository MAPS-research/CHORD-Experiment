#!/bin/bash
#SBATCH --job-name=layer-depth-score
#SBATCH --output=logs/layer_depth/score_%j.out
#SBATCH --error=logs/layer_depth/score_%j.err
#SBATCH --time=6:00:00
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=<CPU_PARTITION>
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
# Table 1 selectivity at every layer of one backbone, after featurize.slurm.
#   SIZE=27b sbatch experiments/layer_depth/score.slurm      # or SIZE=9b
source "${CONDA_ACTIVATE:-conda}" activate chord-gpu 2>/dev/null || conda activate chord-gpu
cd "${CHORD_ROOT:-$(dirname "$0")/../..}"; export PYTHONPATH=$PWD
S=${SIZE:?set SIZE=27b or 9b}
case $S in 27b) ENC=qwen35-27b-prompteol-coherence-l62 ;; 9b) ENC=qwen35-9b-prompteol-coherence-l30 ;; esac
python -m experiments.layer_depth.selectivity_by_layer \
  --sweep-dir outputs/experiments/layer_depth/qwen35_$S --rng-encoder $ENC
