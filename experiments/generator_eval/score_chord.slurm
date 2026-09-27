#!/bin/bash
#SBATCH --job-name=r10-score
#SBATCH --output=logs/gen10/score_%j.out
#SBATCH --error=logs/gen10/score_%j.err
#SBATCH --time=01:00:00
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=<CPU_PARTITION>
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
source "${CONDA_ACTIVATE:-conda}" activate chord-gpu 2>/dev/null || conda activate chord-gpu
cd "${CHORD_ROOT:-$(dirname "$0")/../..}"; export PYTHONPATH=$PWD
ENCS="${ENCODERS:-qwen35-27b-prompteol-coherence-l62 chord-qwen3.5-2b-student chord-qwen3.5-0.8b-student}"
python -m experiments.generator_eval.score_chord_folds --config experiments/generator_eval/configs/featurize.yaml --encoders $ENCS; rc1=$?
python -m experiments.generator_eval.score_chord_folds --config experiments/generator_eval/configs/featurize.yaml --encoders $ENCS --unbiased; rc2=$?
echo "SCORE_EXIT=$rc1/$rc2"; [ $rc1 -eq 0 ] && [ $rc2 -eq 0 ]
