# README Critique

A grouped audit of `README.md` ahead of the public release. Items are roughly ordered by severity within each group.

## Bugs / outdated content (fix before going public)

1. **Line 163** — *"Model type is auto-detected from the HuggingFace config. Encoder vs. causal routing happens automatically in `get_model()`."* — **False.** After the registry refactor, dispatch is prefix-based (`lftk:`, `tfidfngrams:`, `neurobiber:`, …) with `HFModel` as the fallback. There's no HF-config autodetect and the causal-LM branch is gone. The remaining `_is_causal_model` in `core.py` is dead code.

2. **Line 169** — *"Register in `steb/models/__init__.py` by adding to `MODEL_REGISTRY`."* — **False.** `MODEL_REGISTRY` no longer exists; the registry is built lazily inside `get_model_registry()` (lazy so importing one model class doesn't drag in torch/spaCy/LFTK).

3. **Lines 157–158** — example paths `lftk:/STEB/configs/lftk/surface_pos.yaml` and `tfidfngrams:/STEB/models/tfidfngrams_mud_subset_1-2grams.pkl` are **absolute paths to a directory that won't exist for a new user.** Should be relative.

4. **Line 125** — `documentation/hungarian-algorithm.md` link still works, but `documentation/` is a legacy directory holding a single file that's also at `docs/tasks/hungarian-algorithm.md` (the canonical mkdocs path). Pick one and delete the other; the duplicate will rot.

5. **Line 25 vs Line 12** — the installation step is `pip install -e .`, but the note about `download_datasets.sh` says "after `pip install -r requirements.txt`". Either is fine, but be consistent or it reads as a contradiction.

## Missing for a public release

6. **No paper link, no citation, no BibTeX block.** Most-clicked sections of a benchmark README.

7. **No leaderboard / results.** A benchmark README that shows zero results is a missed opportunity. Even a small "Reference numbers for X models" table tells visitors what good looks like.

8. **No badges** (license, paper, Python version, CI). Quick visual cues a public repo is expected to have.

9. **No license callout** (even though `LICENSE` exists).

10. **Line 27** — *"There is also a special dataset download option for datasets requiring licenses or subscriptions, such as datasets from the Linguistic Data Consortium (e.g., Fisher…)"* — teases a feature without telling you how to use it. Either link to instructions or remove.

## Structural

11. **Configuration before Usage is backwards.** A new visitor wants to see "how do I run this" before "here are environment variables for paths I haven't created yet." Move Configuration after Usage (or into an Advanced/Reference section).

12. **No real Quick Start.** Right now the path to a first working invocation is: Installation → Downloading Datasets (with a `--purge` aside and a Note) → Configuration (with a table and an ini-file alternative) → Usage. That's three sections of prerequisites. A `## Quick Start` block immediately after the intro with five lines that get a user to a result, then the deeper sections, is more typical.

13. **The README duplicates a lot of `docs/`.** With mkdocs in place, the README's "Tasks" section (95–145) and "Adding a New Model / Dataset" section (165–212) largely repeat the docs site. Trimming the README to *intro → install → quick start → link to docs* would cut ~50% of the length and remove a second source of truth that can drift (as #1 and #2 above already have).

## Small stuff

14. Intro sentence "modular and extensible" is the kind of generic framing every framework uses. One concrete sentence about *what stylistic property* the benchmark measures, with a one-sentence example, would land harder.

15. **No mention of `steb preview`** in Utility Commands (documented in `docs/usage/cli.md` but not here).

16. **Probing has no CLI example** — every other task in the Tasks section has one.

17. **No mention of `--truncate` / `--max-tokens`** — minor, but it's a real feature that just landed and is worth one line if you want users to use it for fair-comparison runs.

## Recommendation

If fixing this in one PR: (a) fix the three concrete bugs (#1, #2, #3), (b) add a Quick Start + paper/citation block at the top, (c) trim the Tasks and Adding-a-* sections to a short overview + link to docs, and (d) move Configuration to the end. The duplication-with-docs problem (#13) is the biggest long-term maintenance hazard — every doc change becomes a two-edit task and the README will drift like it just did.
