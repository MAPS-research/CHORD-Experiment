#!/bin/bash
#SBATCH --job-name=window-length-score
#SBATCH --output=logs/window_length/score_%j.out
#SBATCH --error=logs/window_length/score_%j.err
#SBATCH --time=12:00:00
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=<CPU_PARTITION>
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
# Fold scores (CHORD z, MAUVE) of every truncation length, then the per-corpus
# CSVs, after window_length_featurize.slurm has run for each length.
#   sbatch experiments/window_length/window_length_score.slurm
source "${CONDA_ACTIVATE:-conda}" activate chord-gpu 2>/dev/null || conda activate chord-gpu
cd "${CHORD_ROOT:-$(dirname "$0")/../..}"; export PYTHONPATH=$PWD
export OMP_NUM_THREADS=4
rc=0
for L in ${LENGTHS:-128 256 384 512 1024}; do
  python -m experiments.window_length.score_folds --dir outputs/experiments/window_length_folds/L$L \
    --workers 8 || rc=1
done
python -m experiments.window_length.collect || rc=1
echo "SCORE_EXIT=$rc"; exit $rc
