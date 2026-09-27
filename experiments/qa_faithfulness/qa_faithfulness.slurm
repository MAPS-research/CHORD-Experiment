#!/bin/bash
#SBATCH --job-name=qa-faith
#SBATCH --output=logs/gen10/qa_%j.out
#SBATCH --error=logs/gen10/qa_%j.err
#SBATCH --time=03:00:00
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
# QA faithfulness (tab:case-qa): featurize the source-conditioned texts, then the
# counterfactual selectivity scorer under the production setting (dev 1500) and a
# dev-150 variant, to find which reproduces the published 27B row.
source "${CONDA_ACTIVATE:-conda}" activate chord-gpu 2>/dev/null || conda activate chord-gpu
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
cd "${CHORD_ROOT:-$(dirname "$0")/../..}"; export PYTHONPATH=$PWD
C=outputs/casestudy/qa_faithfulness/evalset
python -m experiments.counterfactual_eval.featurize --config experiments/counterfactual_eval/configs/encoders_qa_faithfulness.yaml --corpus-dir $C || { echo FEAT_EXIT=1; exit 1; }
python -m experiments.counterfactual_eval.selectivity --config experiments/counterfactual_eval/configs/selectivity.yaml --corpus-dir $C --out-suffix _dev1500; r1=$?
python -m experiments.counterfactual_eval.selectivity --config experiments/counterfactual_eval/configs/selectivity_qa_faithfulness.yaml --corpus-dir $C --out-suffix _dev150; r2=$?
for s in _dev1500 _dev150; do echo "== $s"; awk -F, 'NR==1||$1~/src/{print $1, $2, $5, $6, $11}' $C/selectivity/selectivity_by_family$s.csv; done
echo "QA_EXIT=$r1/$r2"; [ $r1 -eq 0 ] && [ $r2 -eq 0 ]
