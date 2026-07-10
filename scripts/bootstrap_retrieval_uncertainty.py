"""Bootstrap confidence intervals for STEB retrieval MRR (R2's W1c).

Runs the top-5 retrieval models (by mean MRR across the four retrieval
datasets) through the real STEB evaluation harness with per-query
reciprocal ranks retained, then bootstraps confidence intervals on MRR
and on pairwise MRR gaps between models.

This uses the *full* query/target pools (no subsampling), matching the
protocol that produced the paper's published Table 2/6 numbers. Intended
to be run on a CUDA machine; results (including confidence intervals) are
written to --report-json / --report-md so they don't need to be
recomputed or re-summarized by hand.

Usage:
    python -m scripts.bootstrap_retrieval_uncertainty
    python -m scripts.bootstrap_retrieval_uncertainty --batch-size 128 --n-bootstrap 2000

Requires the four retrieval datasets to be present under raw_datasets/
(amazon_retrieval, fanfiction_retrieval, stackexchange_retrieval,
pan18_cross_domain_authorship_attribution) -- see download_datasets.sh.
"""
import argparse
import glob
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from steb.core import get_model, evaluate  # noqa: E402

MODELS = [
    "rrivera1849/LUAR-MUD",
    "rrivera1849/LUAR-CRUD",  # NOTE: scripts/models_all.txt has this as "rivera1849/LUAR-CRUD"
    # (single r), which 401s -- that looks like a typo in the tracked file, not the real repo id.
    "AIDA-UPM/star",
    "Blablablab/multilingual-style-representation",
    "microsoft/deberta-v3-large",
]

DATASETS = [
    "amazon",
    "fanfiction",
    "stackexchange_retrieval",
    "pan18_cross_domain_authorship_attribution_english",
]


def model_short_name(model_name_or_path: str) -> str:
    """Matches the model_str logic in steb.core.evaluate()."""
    short = os.path.basename(model_name_or_path)
    if short == "":
        short = os.path.basename(os.path.dirname(model_name_or_path))
    return short


def find_metrics_json(output_dir: str, dataset: str, model_str: str) -> str:
    pattern = os.path.join(output_dir, dataset, model_str, "**", "retrieval", "metrics.json")
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        raise FileNotFoundError(
            f"No metrics.json found for dataset={dataset} model={model_str} "
            f"under {output_dir}. Did the evaluation run succeed?"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Expected exactly one metrics.json for dataset={dataset} model={model_str}, "
            f"found {len(matches)}: {matches}"
        )
    return matches[0]


def load_per_query_rr(output_dir: str, dataset: str, model_str: str) -> np.ndarray:
    path = find_metrics_json(output_dir, dataset, model_str)
    with open(path) as f:
        metrics = json.load(f)
    if "per_query_rr" not in metrics:
        raise KeyError(
            f"{path} has no 'per_query_rr' field (it was likely computed before this "
            f"script's changes, or with retrieval_return_per_query=False). Rerun with "
            f"--force-rerun to regenerate it."
        )
    return np.array(metrics["per_query_rr"])


def bootstrap_ci(values: np.ndarray, n_bootstrap: int, rng: np.random.Generator, alpha: float = 0.05):
    n = len(values)
    boot_means = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = values[idx].mean()
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(values.mean()), float(lo), float(hi)


def paired_bootstrap_diff(a: np.ndarray, b: np.ndarray, n_bootstrap: int, rng: np.random.Generator, alpha: float = 0.05):
    """Paired bootstrap on mean(a) - mean(b), resampling shared query indices."""
    assert len(a) == len(b), "paired bootstrap requires index-aligned, equal-length arrays"
    n = len(a)
    boot_diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_diffs[i] = a[idx].mean() - b[idx].mean()
    lo, hi = np.percentile(boot_diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    point = float(a.mean() - b.mean())
    significant = bool(lo > 0 or hi < 0)
    return point, float(lo), float(hi), significant


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=MODELS, help="HF model ids to evaluate.")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, help="Retrieval datasets to evaluate on.")
    parser.add_argument("--output-dir", default="results_bootstrap_retrieval",
                         help="Separate results dir so this doesn't touch the main results/ tree.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-rerun", action="store_true",
                         help="Re-run evaluations even if metrics.json already exists "
                              "(needed if a prior run was done without per-query tracking).")
    parser.add_argument("--progress-bar", action="store_true", default=True)
    parser.add_argument("--report-json", default="bootstrap_retrieval_report.json")
    parser.add_argument("--report-md", default="bootstrap_retrieval_report.md")
    args = parser.parse_args()

    print(f"Models: {args.models}")
    print(f"Datasets: {args.datasets}")
    print(f"Output dir: {args.output_dir}")
    print()

    # --- Step 1: run the real evaluation harness per model, retaining per-query MRR ---
    for model_name in args.models:
        print(f"=== Evaluating {model_name} ===")
        model = get_model(model_name)
        evaluate(
            model,
            datasets=args.datasets,
            task_name="retrieval",
            batch_size=args.batch_size,
            output_folder=args.output_dir,
            force_rerun=args.force_rerun,
            progress_bar=args.progress_bar,
            seed=args.seed,
            retrieval_return_per_query=True,
        )
        print()

    # --- Step 2: load per-query RR arrays ---
    per_query = {}  # (model, dataset) -> np.ndarray
    for model_name in args.models:
        model_str = model_short_name(model_name)
        for dataset in args.datasets:
            per_query[(model_name, dataset)] = load_per_query_rr(args.output_dir, dataset, model_str)

    # Sanity check: per-dataset query counts should match across models (same
    # underlying dataset, order determinism assumed from the shared cache).
    for dataset in args.datasets:
        counts = {model_name: len(per_query[(model_name, dataset)]) for model_name in args.models}
        if len(set(counts.values())) > 1:
            print(f"WARNING: query counts differ across models for {dataset}: {counts}. "
                  f"Paired bootstrap comparisons for this dataset will be skipped.")

    # --- Step 3: per-model, per-dataset bootstrap CIs ---
    rng = np.random.default_rng(args.seed)
    ci_results = []
    for model_name in args.models:
        for dataset in args.datasets:
            rr = per_query[(model_name, dataset)]
            mean_mrr, lo, hi = bootstrap_ci(rr, args.n_bootstrap, rng)
            ci_results.append({
                "model": model_name,
                "dataset": dataset,
                "n_queries": len(rr),
                "mrr": mean_mrr,
                "ci_lo": lo,
                "ci_hi": hi,
            })

    # --- Step 4: paired bootstrap for every model pair, per dataset ---
    pair_results = []
    for dataset in args.datasets:
        counts = {model_name: len(per_query[(model_name, dataset)]) for model_name in args.models}
        if len(set(counts.values())) > 1:
            continue
        for model_a, model_b in itertools.combinations(args.models, 2):
            a = per_query[(model_a, dataset)]
            b = per_query[(model_b, dataset)]
            point, lo, hi, significant = paired_bootstrap_diff(a, b, args.n_bootstrap, rng)
            pair_results.append({
                "dataset": dataset,
                "model_a": model_a,
                "model_b": model_b,
                "mrr_diff (a-b)": point,
                "ci_lo": lo,
                "ci_hi": hi,
                "significant_at_95": significant,
            })

    # --- Step 5: write report ---
    report = {
        "config": {
            "models": args.models,
            "datasets": args.datasets,
            "n_bootstrap": args.n_bootstrap,
            "seed": args.seed,
        },
        "per_model_ci": ci_results,
        "pairwise_diff_ci": pair_results,
    }
    with open(args.report_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {args.report_json}")

    lines = ["# Bootstrap retrieval uncertainty report", ""]
    lines.append(f"n_bootstrap={args.n_bootstrap}, seed={args.seed}")
    lines.append("")
    lines.append("## MRR with 95% bootstrap CI")
    lines.append("")
    lines.append("| model | dataset | n_queries | MRR | 95% CI |")
    lines.append("|---|---|---|---|---|")
    for r in ci_results:
        lines.append(
            f"| {model_short_name(r['model'])} | {r['dataset']} | {r['n_queries']} | "
            f"{r['mrr']:.4f} | [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}] |"
        )
    lines.append("")
    lines.append("## Pairwise MRR gap (paired bootstrap, shared resampled queries)")
    lines.append("")
    lines.append("| dataset | model A | model B | MRR(A) - MRR(B) | 95% CI | significant at 95%? |")
    lines.append("|---|---|---|---|---|---|")
    for r in sorted(pair_results, key=lambda r: (r["dataset"], -abs(r["mrr_diff (a-b)"]))):
        lines.append(
            f"| {r['dataset']} | {model_short_name(r['model_a'])} | {model_short_name(r['model_b'])} | "
            f"{r['mrr_diff (a-b)']:+.4f} | [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | "
            f"{'yes' if r['significant_at_95'] else 'no'} |"
        )
    with open(args.report_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {args.report_md}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
