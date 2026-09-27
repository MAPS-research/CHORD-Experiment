#!/bin/bash
#SBATCH --job-name=student-eval
#SBATCH --output=logs/distill/eval_%j.out
#SBATCH --error=logs/distill/eval_%j.err
#SBATCH --time=08:00:00
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
# A trained student's Table 1 row (counterfactual selectivity, merged into the
# existing selectivity CSVs) and Table 2 column (10 x 500 folds, next to the 27B
# and 2B columns), from outputs/distill/student_<STUDENT>/final.
#
#   STUDENT=qwen3.5-0.8b sbatch experiments/student_eval/eval.slurm
#   STUDENT=qwen3.5-2b RUN=student_qwen3.5-2b_rerun sbatch ...   # a retrained copy: outputs/distill/$RUN/final,
#                                                  encoder chord-qwen3.5-2b-student-rerun
source "${CONDA_ACTIVATE:-conda}" activate chord-gpu 2>/dev/null || conda activate chord-gpu
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=16
cd "${CHORD_ROOT:-$(dirname "$0")/../..}"; export PYTHONPATH=$PWD
S=${STUDENT:?set STUDENT=qwen3.5-2b or qwen3.5-0.8b}; RUN=${RUN:-student_$S}
ENC=chord-$S-student; [ "$RUN" != "student_$S" ] && ENC=chord-$S-student-${RUN#student_${S}_}
CKPT=$PWD/outputs/distill/$RUN/final; CF=outputs/counterfactual/meta_eval
UG=experiments/generator_eval/configs
[ -f "$CKPT/projector.pt" ] || { echo "no trained checkpoint at $CKPT"; exit 1; }
# point the student's encoder block at this checkpoint and name (the tracked
# configs carry a placeholder or a local path; the featurize copy stays in its
# directory so its relative paths resolve there)
TMP_CF=$(mktemp); TMP_UG=$UG/.featurize_$RUN.$$.yaml
trap 'rm -f "$TMP_CF" "$TMP_UG"' EXIT
SUB=(-e "s|model: .*outputs/distill/student_$S/final\$|model: $CKPT|" -e "s|name: chord-$S-student\$|name: $ENC|")
sed "${SUB[@]}" experiments/counterfactual_eval/configs/encoders_chord_$S.yaml > "$TMP_CF"
sed "${SUB[@]}" $UG/featurize.yaml > "$TMP_UG"
grep -q "model: $CKPT" "$TMP_CF" && grep -q "name: $ENC" "$TMP_UG" || { echo "checkpoint not substituted"; exit 1; }
set -e
nvidia-smi --query-gpu=name,memory.total --format=csv
python -m experiments.counterfactual_eval.featurize --config "$TMP_CF" --corpus-dir $CF
python -m experiments.counterfactual_eval.selectivity --config experiments/counterfactual_eval/configs/selectivity.yaml --corpus-dir $CF --only-encoders $ENC
python -m chord.featurize --config "$TMP_UG" --only-encoder $ENC
ENCS=$(printf '%s\n' qwen35-27b-prompteol-coherence-l62 chord-qwen3.5-2b-student $ENC | awk '!seen[$0]++')
python -m experiments.generator_eval.score_chord_folds --config $UG/featurize.yaml --encoders $ENCS
python -m experiments.generator_eval.score_chord_folds --config $UG/featurize.yaml --encoders $ENCS --unbiased
echo "STUDENT_EVAL_EXIT=0"
