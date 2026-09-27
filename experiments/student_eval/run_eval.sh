#!/usr/bin/env bash
# =============================================================================
# Evaluate a distilled student on the two lanes the paper reports it on, then
# print the comparison against the 27B teacher.
#
#   experiments/student_eval/run_eval.sh <encoder-name> [checkpoint-dir]
#
# Example:
#   experiments/student_eval/run_eval.sh my-student-v1 outputs/distill/student_qwen3.5-2b/final
#
# What it runs:
#   1. counterfactual featurize            (GPU)  -> outputs/counterfactual/meta_eval/feats/<name>/
#   2. prompt_split_eval            (CPU)  -> honest selection/confirmation split
#   3. selectivity                  (CPU)  -> full evaluation-set per-condition z
#   4. unconditional-gen featurize  (GPU)  -> outputs/distill/eval_ladder/features/<name>/
#   5. score_chord             (CPU)  -> raw RBF-MMD ladder
#   6. experiments.student_eval.compare        (CPU)  -> the two paper-format tables
#
# Students are judged on the CONFIRMATION half of step 2, never on the full
# counterfactual: the student's training touched this protocol, so the selection half
# is the only half that may inform any choice, and the confirmation half is what
# gets reported. Step 3 is still useful context but is not the reported number.
#
# Set PYTHON_GPU / PYTHON_CPU if your GPU and CPU stacks live in different envs
# (the reference setup uses chord-gpu and chord-cpu).
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
export TOKENIZERS_PARALLELISM=false

NAME="${1:-}"
CKPT="${2:-outputs/distill/student_qwen3.5-2b/final}"
if [ -z "$NAME" ]; then
  sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 2
fi
# The name becomes an encoder id, a directory name and a sed replacement, so
# keep it to characters that are safe in all three.
if ! [[ "$NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "FATAL: encoder name must match [A-Za-z0-9._-]+ (got '$NAME')" >&2
  exit 2
fi

PY_GPU="${PYTHON_GPU:-${PYTHON:-python}}"
PY_CPU="${PYTHON_CPU:-${PYTHON:-python}}"

COUNTERFACTUAL_DIR="${COUNTERFACTUAL_DIR:-outputs/counterfactual/meta_eval}"
TEACHER_LADDER="${TEACHER_LADDER:-outputs/casestudy/single_fold_chord_27b}"
LADDER_DIR=outputs/distill/eval_ladder
SUFFIX="_${NAME//[^A-Za-z0-9]/_}"

if [ ! -d "$CKPT" ]; then
  echo "FATAL: checkpoint dir not found: $CKPT" >&2
  exit 1
fi
if [ ! -f "$COUNTERFACTUAL_DIR/conditions_manifest.json" ]; then
  echo "FATAL: no counterfactual at $COUNTERFACTUAL_DIR (build it, or unpack the data bundle)" >&2
  exit 1
fi

# Materialize per-run configs with this encoder name + checkpoint, so nothing
# depends on the user having hand-edited the tracked templates.
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
CKPT_ABS="$(cd "$CKPT" && pwd)"

sed -e "s|name: my-student-v1|name: ${NAME}|" \
    -e "s|model: /ABSOLUTE/PATH/TO/outputs/distill/student_qwen3.5-2b/final|model: ${CKPT_ABS}|" \
    experiments/student_eval/configs/eval_student_counterfactual.yaml > "$WORK/counterfactual.yaml"

# the ladder config lives in experiments/student_eval/configs/, so relative paths inside it are
# resolved against that directory — keep it there and only swap the two fields.
# The filename carries $NAME so two evaluations can run concurrently: with a
# shared name, one job's exit trap deletes the file out from under the other,
# and a later reader picks up the wrong encoder/checkpoint pair.
sed -e "s|name: my-student-v1|name: ${NAME}|" \
    -e "s|mmd_encoder: my-student-v1|mmd_encoder: ${NAME}|" \
    -e "s|model: /ABSOLUTE/PATH/TO/outputs/distill/student_qwen3.5-2b/final|model: ${CKPT_ABS}|" \
    experiments/student_eval/configs/eval_student_single_fold.yaml > experiments/student_eval/configs/.eval_ladder_run_${NAME}.yaml
trap 'rm -rf "$WORK"; rm -f experiments/student_eval/configs/.eval_ladder_run_${NAME}.yaml' EXIT

echo "=== [1/6] counterfactual featurize (GPU) ==="
"$PY_GPU" -m experiments.counterfactual_eval.featurize --config "$WORK/counterfactual.yaml" --corpus-dir "$COUNTERFACTUAL_DIR"

echo "=== [2/6] honest split evaluation (CPU) ==="
"$PY_CPU" -m experiments.counterfactual_eval.prompt_split_eval --corpus-dir "$COUNTERFACTUAL_DIR" --draws 80 \
  --encoders qwen35-27b-prompteol-coherence-l62 "$NAME" --out-suffix "$SUFFIX"

echo "=== [3/6] full evaluation-set selectivity (CPU) ==="
"$PY_CPU" -m experiments.counterfactual_eval.selectivity --config experiments/counterfactual_eval/configs/selectivity.yaml \
  --corpus-dir "$COUNTERFACTUAL_DIR" --only-encoders "$NAME"

echo "=== [4/6] unconditional-generation featurize (GPU) ==="
"$PY_GPU" -m chord.featurize --config experiments/student_eval/configs/.eval_ladder_run_${NAME}.yaml \
  --only-encoder "$NAME"

echo "=== [5/6] unconditional-generation scoring (CPU) ==="
"$PY_CPU" -m experiments.generation_scoring.score_chord --config experiments/student_eval/configs/.eval_ladder_run_${NAME}.yaml \
  --encoders "$NAME"

echo "=== [6/6] comparison against the 27B teacher ==="
"$PY_CPU" -m experiments.student_eval.compare --student "$NAME" \
  --counterfactual-dir "$COUNTERFACTUAL_DIR" --split-suffix "$SUFFIX" \
  --ladder-dir "$LADDER_DIR" --teacher-ladder-dir "$TEACHER_LADDER" --extras
