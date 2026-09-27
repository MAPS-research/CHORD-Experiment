#!/usr/bin/env bash
# =============================================================================
# Counterfactual meta-evaluation set, end to end:
#
#   build the evaluation set -> data-quality gate -> featurize -> selectivity -> dose-response
#
# Usage:
#   experiments/counterfactual_eval/run_meta_eval.sh          # Qwen3.5-27B backbone (one large GPU)
#
# Expects the 3-domain corpus (see scripts/download_openwebtext.py and the
# README) and, for the LLM-edited counterfactual families, the
# perturbation file produced by chord.data.generate_perturbations (needs an
# OpenAI-compatible editor endpoint; see experiments/counterfactual_eval/configs/perturbations_qwen_editor.yaml).
# Missing optional inputs are skipped with a log line, not silently.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
export TOKENIZERS_PARALLELISM=false

PY="${PYTHON:-python}"

PREP_CFG=experiments/counterfactual_eval/configs/select_source_passages.yaml
BUILD_CFG=experiments/counterfactual_eval/configs/build_evalset.yaml
ENC_CFG=experiments/counterfactual_eval/configs/encoders_chord_27b.yaml
SEL_CFG=experiments/counterfactual_eval/configs/selectivity.yaml
CORPUS=outputs/counterfactual/meta_eval

echo "[run_meta_eval] corpus=$CORPUS backbone=$ENC_CFG"

echo "[run_meta_eval] 0/5 prepare 3-domain corpus (CPU)"
"$PY" -m chord.data.split_passages --config "$PREP_CFG"

echo "[run_meta_eval] 1/5 build evaluation set (CPU)"
"$PY" -m chord.data.counterfactual --config "$BUILD_CFG"

echo "[run_meta_eval] 2/5 data-quality inspection (CPU) -- GATE"
"$PY" -m experiments.counterfactual_eval.inspect_corpus --corpus-dir "$CORPUS" --samples 5

echo "[run_meta_eval] 3/5 featurize (GPU)"
"$PY" -m experiments.counterfactual_eval.featurize --config "$ENC_CFG" --corpus-dir "$CORPUS"

echo "[run_meta_eval] 4/5 selectivity (CPU)"
"$PY" -m experiments.counterfactual_eval.selectivity --config "$SEL_CFG" --corpus-dir "$CORPUS"

echo "[run_meta_eval] 5/5 dose-response / monotonicity (CPU)"
"$PY" -m experiments.counterfactual_eval.dose_response --corpus-dir "$CORPUS"

echo "[run_meta_eval] DONE -> $CORPUS/selectivity/ + $CORPUS/dose_response/ + $CORPUS/qc/"
