#!/bin/bash
#SBATCH --job-name=r10-cost
#SBATCH --output=logs/gen10/cost_%j.out
#SBATCH --error=logs/gen10/cost_%j.err
#SBATCH --time=03:00:00
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>   # measured on one H200 (see the appendix)

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
# Compute-cost appendix: every row in one process on one GPU (whichever of the
# two submitted partitions starts first; the other is canceled).
source "${CONDA_ACTIVATE:-conda}" activate chord-gpu 2>/dev/null || conda activate chord-gpu
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
cd "${CHORD_ROOT:-$(dirname "$0")/../..}"; export PYTHONPATH=$PWD
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -m experiments.compute_cost.bench_cost --out outputs/casestudy/unconditional_generation/scores/cost.csv
rc=$?; echo "COST_EXIT=$rc"; exit $rc
