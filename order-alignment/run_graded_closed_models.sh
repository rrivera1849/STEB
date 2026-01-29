#!/bin/sh

### Job name
#SBATCH --job-name=eval_graded_closed_models

### Where STDOUT/STDERR will be written (%j = job id)
#SBATCH -o order-alignment/slurm_eval_graded_closed_models_%j.txt

### Walltime [hour:]minute:second
#SBATCH -t 30:00:00

### Memory per node
#SBATCH --mem 8G

# Always run from the submission directory (repo root if submitted there)
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# ---- Python venv (.venv is one folder above repo root) ----
# If you submit from the repo root, this activates ../.venv
. "venv/bin/activate"
export OPENAI_API_KEY="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"

# Output directory (one CSV per model will be written here)
OUT_DIR="order-alignment/graded_closed_model_outputs/${SLURM_JOB_ID:-manual}"

python order-alignment/eval_graded_closed_models.py \
  --output-dir "${OUT_DIR}" \
  --max-new-tokens 512 \
  --temperature 1.0 \
  --seed 42 \
  --full-run \
  --model "gpt-5-mini-2025-08-07"
