<!--
PR template for community model submissions to STEB.
See the "Submitting your model" section of README.md for the full flow.
-->

## Model submission

**Short name:** `<short_name>`
**HF id:** `<org/model>`
**Run command:** `<exact CLI used>`

## Checklist

- [ ] Ran `python -m scripts.benchmark_clustering` locally; the new model appears in the `STEB_operational` and `STEB_definitional` sheets of `scores.xlsx`.
- [ ] Added a 4-key entry to `SUBMISSIONS.yaml`.
- [ ] Added results to `submitted_results/<dataset>/<short_name>/...` for every dataset evaluated.
- [ ] Added `org/<short_name>` to `scripts/models_all.txt`.
- [ ] Ran `python scripts/validate_submission.py` locally; output is clean.

## Notes (optional)

Anything that would help reviewers — training data summary, hardware used, sanity-check comparisons against published numbers, license caveats, etc.
