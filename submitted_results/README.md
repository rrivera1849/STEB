# Community model submissions

This directory holds STEB benchmark results contributed by the community. The maintainer-owned `results/` (typically `STEB_RESULTS_DIR`) stays separate so PR review and `git log` blame stay clean.

## Layout

Mirror the standard STEB results layout, with the contributor's `short_name` as the model-level directory:

```
submitted_results/
  <dataset>/
    <short_name>/
      <episode_config>/
        <task>/
          metrics.json
```

The standard `python -m scripts.benchmark_clustering` invocation auto-merges this tree with the maintainer's `--results-dir` on every run. On collisions (same dataset / model / episode config / task), the maintainer-owned numbers win.

## How to contribute

See [`SUBMISSION.md`](../SUBMISSION.md) at the repo root.
