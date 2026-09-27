#!/bin/bash
#SBATCH --job-name=window-length-feat
#SBATCH --output=logs/window_length/featurize_%j.out
#SBATCH --error=logs/window_length/featurize_%j.err
#SBATCH --time=12:00:00
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
# CHORD 27B and MAUVE GPT-2-large features of one truncation length L (appendix
# "Sensitivity to Evaluation Window Length"), after build_truncated_corpora.py.
# featurize caches per file, so a requeued job resumes where it stopped.
#   L=512 sbatch experiments/window_length/window_length_featurize.slurm
source "${CONDA_ACTIVATE:-conda}" activate chord-gpu 2>/dev/null || conda activate chord-gpu
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} TOKENIZERS_PARALLELISM=false
cd "${CHORD_ROOT:-$(dirname "$0")/../..}"; export PYTHONPATH=$PWD
nvidia-smi --query-gpu=name,memory.total --format=csv
D=outputs/experiments/window_length_folds/L${L:?set L to the truncation length}
rc=0
python -m chord.featurize --config $D/chord.yaml || rc=1
python -m chord.featurize --config $D/mauve.yaml || rc=1
echo "FEAT_EXIT=$rc"; exit $rc
