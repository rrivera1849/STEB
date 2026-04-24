#!/bin/bash
set -euo pipefail

# Usage: ./scripts/launch_benchmark.sh <models_file>
# Reads one HuggingFace model ID per line and submits a SLURM job for each.

if [ $# -lt 1 ]; then
    echo "Usage: $0 <models_file>"
    echo "  models_file: plain-text file with one HF model ID per line"
    exit 1
fi

MODELS_FILE="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="${PROJECT_DIR}/slurm_logs"
mkdir -p "$LOG_DIR"

while IFS= read -r model || [ -n "$model" ]; do
    # Skip blank lines and comments
    model="$(echo "$model" | xargs)"
    [[ -z "$model" || "$model" == \#* ]] && continue

    # Sanitize model name for job name and log files
    job_name="steb_$(echo "$model" | tr '/' '_')"

    echo "Submitting: $model"
    sbatch \
        --job-name="$job_name" \
        --output="${LOG_DIR}/${job_name}_%j.out" \
        --error="${LOG_DIR}/${job_name}_%j.err" \
        --nodes=1 \
        --gpus=1 \
        --time=24:00:00 \
        --wrap="cd ${PROJECT_DIR} && steb --preset benchmark \"${model}\" --progress-bar"

done < "$MODELS_FILE"

echo "All jobs submitted."
