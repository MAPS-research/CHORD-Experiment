#!/bin/bash
#SBATCH --job-name=r10-feat-students
#SBATCH --output=logs/gen10/feat_students_%j.out
#SBATCH --error=logs/gen10/feat_students_%j.err
#SBATCH --time=04:00:00
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
# featurize caches per run file, so a requeued job resumes where it stopped
source "${CONDA_ACTIVATE:-conda}" activate chord-gpu 2>/dev/null || conda activate chord-gpu
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} TOKENIZERS_PARALLELISM=false
cd "${CHORD_ROOT:-$(dirname "$0")/../..}"; export PYTHONPATH=$PWD
nvidia-smi --query-gpu=name,memory.total --format=csv
rc=0
for ENC in ${ENCODERS:-chord-2b-student chord-qwen3.5-0.8b-student}; do
  python -m chord.featurize --config experiments/generator_eval/configs/featurize.yaml --only-encoder $ENC || rc=1
done
echo "FEAT_EXIT=$rc"; exit $rc
