#!/bin/sh

### Job name
#SBATCH --job-name=eval_graded_open_models

### Where STDOUT/STDERR will be written (%j = job id)
#SBATCH -o order-alignment/slurm_eval_graded_open_models_%j.txt

### Walltime [hour:]minute:second
#SBATCH -t 40:00:00

### Request a full A100 GPU (see your cluster docs)
#SBATCH -p gpu --gpus-per-node=7g.79gb:1

### Memory per node
#SBATCH --mem 70G

# Always run from the submission directory (repo root if submitted there)
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# ---- Python venv (.venv is one folder above repo root) ----
# If you submit from the repo root, this activates ../.venv
. "venv/bin/activate"

# Output directory (one JSON per model will be written here)
OUT_DIR="order-alignment/graded_open_model_outputs/${SLURM_JOB_ID:-manual}"

python order-alignment/eval_graded_open_models.py \
  --output-dir "${OUT_DIR}" \
  --max-new-tokens 512 \
  --temperature 1.0 \
  --seed 42

