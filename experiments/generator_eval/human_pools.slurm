#!/bin/bash
#SBATCH --job-name=r10-human
#SBATCH --output=logs/gen10/human_%j.out
#SBATCH --error=logs/gen10/human_%j.err
#SBATCH --time=03:00:00
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=<CPU_PARTITION>
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
source "${CONDA_ACTIVATE:-conda}" activate chord-gpu 2>/dev/null || conda activate chord-gpu
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} TOKENIZERS_PARALLELISM=false
cd "${CHORD_ROOT:-$(dirname "$0")/../..}"; export PYTHONPATH=$PWD
python -m experiments.generator_eval.build_human_pools --out outputs/casestudy/unconditional_generation/human
rc=$?; echo "HUMAN_EXIT=$rc"; exit $rc
