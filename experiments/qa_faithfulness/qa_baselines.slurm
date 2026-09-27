#!/bin/bash
#SBATCH --job-name=qa-baselines
#SBATCH --output=logs/gen10/qabase_%j.out
#SBATCH --error=logs/gen10/qabase_%j.err
#SBATCH --time=01:30:00
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
# QA faithfulness baselines (GPT-2-large last token, NeoBERT masked mean, BERT
# masked mean) under the reproducing protocol (dev 150, n = 462, B = 200).
source "${CONDA_ACTIVATE:-conda}" activate chord-gpu 2>/dev/null || conda activate chord-gpu
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
cd "${CHORD_ROOT:-$(dirname "$0")/../..}"; export PYTHONPATH=$PWD
C=outputs/casestudy/qa_faithfulness/evalset
rc=0
for E in gpt2-large-src-last neobert-src-mean bert-src-mean; do
  python -m experiments.counterfactual_eval.featurize --config experiments/counterfactual_eval/configs/encoders_qa_faithfulness.yaml --corpus-dir $C --only-encoder $E || { echo "FEAT_FAIL $E"; rc=1; continue; }
  python -m experiments.counterfactual_eval.selectivity --config experiments/counterfactual_eval/configs/selectivity_qa_faithfulness.yaml --corpus-dir $C --out-suffix _dev150 --only-encoders $E || rc=1
done
awk -F, 'NR==1||$1~/gpt2|neobert|bert-src/{print $1, $2, $5, $11}' $C/selectivity/selectivity_by_family_dev150.csv
echo "QABASE_EXIT=$rc"; exit $rc
