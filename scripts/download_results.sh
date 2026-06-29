#!/bin/sh

# Download the canonical STEB benchmark results from the maintainer's
# Google Drive. The results tree is needed to:
#   - Regenerate the Excel workbook (`python -m scripts.benchmark_clustering`).
#   - Reproduce the public leaderboard locally.
#   - Compare a newly-evaluated model against the existing baselines.
#
# Usage: ./scripts/download_results.sh [--purge] [output_dir]
#   --purge      Remove the existing results/ tree before downloading.
#   output_dir   Directory to extract into (default: ./results).

set -e

# TODO: paste the Google Drive file ID for the results.tar.gz archive here.
# The script targets a *single tarball* rather than a folder of JSON files
# because gdown's folder mode hits Drive's rate limits with the ~15k files
# in this tree, and a tarball compresses 62 MB of JSON down to a single,
# fast download.
RESULTS_FILE_ID="https://drive.google.com/file/d/1EC-PLHlMdvS-eKRC8bBfQLOGNg-wiRC7/view?usp=sharing"

PURGE=false
OUTPUT_DIR=""

for arg in "$@"; do
    if [ "$arg" = "--purge" ]; then
        PURGE=true
    else
        OUTPUT_DIR="$arg"
    fi
done

OUTPUT_DIR="${OUTPUT_DIR:-./results}"

if [ "$RESULTS_FILE_ID" = "REPLACE_ME" ]; then
    echo "ERROR: RESULTS_FILE_ID is unset in scripts/download_results.sh." >&2
    echo "Edit the script and paste the Google Drive file ID for results.tar.gz." >&2
    exit 1
fi

if [ "$PURGE" = "true" ]; then
    echo "Purging existing ${OUTPUT_DIR}..."
    rm -rf "$OUTPUT_DIR"
fi

if [ -d "$OUTPUT_DIR" ] && [ -n "$(ls -A "$OUTPUT_DIR" 2>/dev/null)" ]; then
    echo "${OUTPUT_DIR}/ already exists and is non-empty."
    echo "Pass --purge to force a fresh download."
    exit 0
fi

mkdir -p "$(dirname "$OUTPUT_DIR")"

TMPDIR_LOCAL="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_LOCAL"' EXIT

echo "Downloading canonical STEB results from Google Drive..."
gdown "$RESULTS_FILE_ID" -O "$TMPDIR_LOCAL/results.tar.gz" --fuzzy

echo "Extracting into ${OUTPUT_DIR}..."
mkdir -p "$OUTPUT_DIR"
tar -xzf "$TMPDIR_LOCAL/results.tar.gz" -C "$(dirname "$OUTPUT_DIR")"

echo "Done. Canonical results are in ${OUTPUT_DIR}/."
