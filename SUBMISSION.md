# Submitting a model to STEB

If you've evaluated a new model with STEB and want it on the [public leaderboard](https://rrivera1849.github.io/STEB/leaderboard/), open a PR with three small changes. The whole submission is usually a four-line metadata block plus the JSON files STEB already wrote when you ran the benchmark.

Inspired by — and intentionally lighter than — [MTEB's contribution flow](https://github.com/embeddings-benchmark/mteb): no per-model Python wrapper, no HuggingFace dataset, no leaderboard re-fit on the maintainer side.

## The contributor workflow

Five steps, ~10 minutes once your model is evaluated.

### 1. Get the canonical baseline results

The leaderboard's headline numbers are computed against ~40 baseline models the maintainers have already evaluated. Download them once:

```bash
./scripts/download_results.sh
```

This pulls a single tarball (~10 MB compressed) from the maintainer's Google Drive and extracts it into `./results/`. The first time it runs you'll need `gdown` (already in `requirements.txt`). Pass `--purge` to force a fresh download.

### 2. Run STEB on your model, into `submitted_results/`

Point `STEB_RESULTS_DIR` at the community tree so your model's metrics land directly where the contribution belongs, instead of mixing into the canonical baselines:

```bash
export STEB_RESULTS_DIR=./submitted_results
steb "your-org/your-model"
```

This writes `submitted_results/<dataset>/<short_name>/<episode_config>/<task>/metrics.json` for every dataset/task combination.

`<short_name>` is the part after the last `/` in your HF id (e.g. `your-model` for `your-org/your-model`). It's the directory name STEB picks automatically.

### 3. Regenerate the Excel workbook

```bash
unset STEB_RESULTS_DIR     # so benchmark_clustering reads ./results/ by default
python -m scripts.benchmark_clustering
```

`benchmark_clustering` auto-merges `./submitted_results/` with `./results/` (maintainer baselines win on the rare collision), so the resulting `scores.xlsx` shows your model alongside every other. Sanity-check the `STEB_operational` and `STEB_definitional` sheets — your model should appear with reasonable numbers.

### 4. Declare the submission

Three small edits in the repo:

- **`SUBMISSIONS.yaml`** — append a 4-key entry:

    ```yaml
    - short_name: <your-short-name>
      hf_id: <your-org/your-model>
      run_command: steb "<your-org/your-model>"
      contributor: <your-github-handle>
    ```

- **`scripts/models_all.txt`** — add `<your-org>/<your-short-name>` (one new line).

- (Already done in step 2:) the `submitted_results/<dataset>/<your-short-name>/...` subtree is in place.

### 5. Validate and open the PR

```bash
python scripts/validate_submission.py
```

If it prints `OK — all checks passed`, open the PR with the model_submission template. CI runs the same validator on every PR that touches `SUBMISSIONS.yaml`, `submitted_results/`, or `scripts/models_all.txt`.

## `SUBMISSIONS.yaml` schema

Four required keys per entry. Order in the file doesn't matter; entries are append-only.

```yaml
- short_name: LUAR-MUD                       # Must match the directory name under submitted_results/
  hf_id: rrivera1849/LUAR-MUD                # HuggingFace identifier (or a stable URL if not on HF)
  run_command: steb "rrivera1849/LUAR-MUD"   # Exact CLI used to produce the metrics files
  contributor: rrivera1849                   # Your GitHub handle
```

That's the entire schema. Anything beyond these four keys is a future addition; don't pre-emptively invent fields.

## Why so lightweight

We considered a richer per-model YAML (paper, license, training data, embedding dim, parameter count, hardware, …) but explicitly decided not to require it in the first pass:

- **Friction kills contributions.** Every required field is a chance for a would-be contributor to give up.
- **The benchmark numbers are the point.** Metadata is supporting evidence; the scores tell the story.
- **Additive migration is free.** If we later want to filter the leaderboard by license or training-data type, we add an optional field and backfill the entries that care. No schema migration, no broken PRs.

If you want to volunteer extra context, put it in the PR description rather than the YAML.

## What the validator checks (`scripts/validate_submission.py`)

1. `SUBMISSIONS.yaml` parses, and every entry has the four required keys with non-empty string values.
2. Every entry's `short_name` corresponds to at least one results subtree under `submitted_results/<dataset>/<short_name>/...`.
3. Every metrics file under `submitted_results/` parses as JSON.
4. Every entry's `hf_id` (or the bare `short_name`) is present in `scripts/models_all.txt`.

The validator returns non-zero on any failure and prints a copy-pasteable summary suitable for a PR comment.

## After your PR merges — how the leaderboard refreshes

The [public leaderboard](https://rrivera1849.github.io/STEB/leaderboard/) is a static Markdown page (`docs/leaderboard.md`) generated from `scores.xlsx`. The maintainer refresh recipe:

```bash
./scripts/download_results.sh             # ensures ./results/ is up-to-date (skips if present)
python -m scripts.benchmark_clustering    # regenerates scores.xlsx, auto-merging submitted_results/
python scripts/build_leaderboard.py       # rewrites docs/leaderboard.md
git commit -am "Refresh leaderboard"
git push                                   # triggers the docs deploy → GH Pages
```

A maintainer typically does this after merging a batch of submissions, not after every single one. A future enhancement will run this in CI on push-to-main so the refresh becomes automatic; the path is open now that the canonical results are reachable by a download.

## Decisions log

| Question | Decision |
|---|---|
| Single results tree, or split? | Split: `results/` is maintainer-owned (downloaded via `download_results.sh`), `submitted_results/` is community. |
| Auto-ingest `submitted_results/`? | Yes — `benchmark_clustering` reads it whenever it exists. |
| Required vs optional metadata fields? | All four `SUBMISSIONS.yaml` keys are required; no optional fields in Level 1. |
| CI validation? | Yes — GitHub Actions runs `validate_submission.py` on PRs touching `SUBMISSIONS.yaml`, `submitted_results/`, or `scripts/models_all.txt`. |
| Where do the canonical results live? | Google Drive tarball, downloaded via `scripts/download_results.sh`. Not committed to git (62 MB / ~15k JSON files). |
| Leaderboard surface? | Static Markdown rendered via mkdocs to GitHub Pages, refreshed by a maintainer running `build_leaderboard.py` (auto-refresh in CI is the next iteration). |
| Restrictive model licenses for metadata redistribution? | Open: assumed acceptable for Level 1. |

## What's deliberately not in scope yet

- **Auto-refresh of the leaderboard on every submission merge.** Manual today; the gdown-based download script unblocks a future CI workflow that does `download_results.sh` → `benchmark_clustering` → `build_leaderboard.py` → commit.
- **Versioning of the benchmark suite.** If we add or remove datasets later, older submissions may be missing coverage. No re-run policy or version pinning yet.
- **Maintainer re-runs of contributor numbers.** The validator's checks plus the auto-included `submitted_results/` ingestion are the only guards against bad numbers. Spot-checks are case-by-case.
- **License taxonomy.** Apache-2.0 redistribution of bare benchmark scores is treated as acceptable for any model license; if you disagree, open an issue.
