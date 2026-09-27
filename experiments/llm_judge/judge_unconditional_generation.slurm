#!/bin/bash
#SBATCH --job-name=judge-uncond-folds
#SBATCH --output=logs/llm_judge/judge_uncond_folds_%j.out
#SBATCH --error=logs/llm_judge/judge_uncond_folds_%j.err
#SBATCH --time=12:00:00
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
# Appendix table tab:judge-casestudy on Table 2's folds: serve the Qwen3.5-27B
# judge with vLLM, rate every fold text, stop the server. Ratings are cached per
# text, so a requeued job resumes where it stopped.
#   VLLM_PY=<python of an env with vllm> sbatch experiments/llm_judge/judge_unconditional_generation.slurm
source "${CONDA_ACTIVATE:-conda}" activate chord-gpu 2>/dev/null || conda activate chord-gpu
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} TOKENIZERS_PARALLELISM=false
cd "${CHORD_ROOT:-$(dirname "$0")/../..}"; export PYTHONPATH=$PWD
nvidia-smi --query-gpu=name,memory.total --format=csv
VLLM_PY=${VLLM_PY:?set VLLM_PY to the python of an environment with vllm installed}
# Triton backends avoid JIT-compiling FlashInfer; vLLM's own kernels need its `ninja`,
# so the vllm env goes on PATH for the server only (the judge runs in chord-gpu)
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=TRITON_ATTN
LOG=logs/llm_judge/vllm_${SLURM_JOB_ID:-local}.log; mkdir -p logs/llm_judge
PATH="$(dirname "$VLLM_PY"):$PATH" "$VLLM_PY" -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3.5-27B \
  --revision fc05daec18b0a78c049392ed2e771dde82bdf654 --served-model-name qwen35-27b-judge \
  --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.90 --dtype bfloat16 \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' > "$LOG" 2>&1 &
PID=$!
for _ in $(seq 1 120); do
  curl -s http://127.0.0.1:8000/v1/models | grep -q qwen35-27b-judge && break
  sleep 10
done
if ! curl -s http://127.0.0.1:8000/v1/models | grep -q qwen35-27b-judge; then
  echo "judge server did not come up; last log lines:"; tail -40 "$LOG"; kill $PID; exit 4
fi
python -m experiments.llm_judge.judge_unconditional_generation \
  --config experiments/llm_judge/configs/judge_unconditional_generation.yaml
rc=$?
kill $PID 2>/dev/null; wait $PID 2>/dev/null
echo "JUDGE_EXIT=$rc"; exit $rc
