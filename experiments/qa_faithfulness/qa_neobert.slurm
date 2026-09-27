#!/bin/bash
#SBATCH --job-name=qa-neobert
#SBATCH --output=logs/gen10/qaneo_%j.out
#SBATCH --error=logs/gen10/qaneo_%j.err
#SBATCH --time=01:00:00
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# NeoBERT's remote modeling code hard-depends on xformers, which is not part of
# the chord-gpu stack; featurize in a side env (conda env `neobert`: torch,
# xformers, transformers), then score in chord-gpu.
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
cd "${CHORD_ROOT:-$(dirname "$0")/../..}"; export PYTHONPATH=$PWD
C=outputs/casestudy/qa_faithfulness/evalset
conda activate neobert
python -m experiments.counterfactual_eval.featurize --config experiments/counterfactual_eval/configs/encoders_qa_faithfulness.yaml --corpus-dir $C --only-encoder neobert-src-mean || { echo FEAT_EXIT=1; exit 1; }
conda activate chord-gpu
python -m experiments.counterfactual_eval.selectivity --config experiments/counterfactual_eval/configs/selectivity_qa_faithfulness.yaml --corpus-dir $C --out-suffix _dev150 --only-encoders neobert-src-mean; rc=$?
awk -F, 'NR==1||$1~/neobert/{print $1, $2, $5, $11}' $C/selectivity/selectivity_by_family_dev150.csv
echo "QANEO_EXIT=$rc"; exit $rc
