#!/bin/bash
#SBATCH --job-name=r10-baselines
#SBATCH --output=logs/gen10/baselines_%j.out
#SBATCH --error=logs/gen10/baselines_%j.err
#SBATCH --time=05:00:00
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
# Table-2 baseline columns under the 10x500 fold protocol: MAUVE-GPT2 features
# (cached per file, requeue-safe), then gen-PPL / entropy / MAUVE per fold.
source "${CONDA_ACTIVATE:-conda}" activate chord-gpu 2>/dev/null || conda activate chord-gpu
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=16
cd "${CHORD_ROOT:-$(dirname "$0")/../..}"; export PYTHONPATH=$PWD
nvidia-smi --query-gpu=name --format=csv,noheader
python -m chord.featurize --config experiments/generator_eval/configs/featurize.yaml --only-encoder mauve-gpt2 || { echo FEAT_EXIT=1; exit 1; }
python -m experiments.generator_eval.score_baselines_folds --config experiments/generator_eval/configs/featurize.yaml
rc=$?; echo "BASELINES_EXIT=$rc"; exit $rc
