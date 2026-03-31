"""Generate dummy results to test benchmark_clustering.py.

Creates synthetic metrics.json files for 15 fake models across the clustering
task's 19 datasets. Scores are designed so that datasets form ~3 recognizable
clusters based on how they rank models.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from steb.core import get_supported_datasets

DUMMY_DIR = "dummy_results"
N_MODELS = 15
SEED = 42

# Clustering task datasets and their metric
TASK = "clustering"
PATH_TASK = "clustering"
METRIC = "v_measure"
EPISODE_PARAMS = "1_50"

# We'll create 3 "archetypes" — groups of datasets that rank models similarly.
# Within each group, datasets share a base ranking with added noise.
# Across groups, the rankings are shuffled differently.
CLUSTER_GROUPS = {
    "topic_classification": [
        "20_Newsgroups_Fixed",
        "ag_news",
        "reuters21578",
        "emotion",
        "twitter-airline-sentiment",
        "financial_phrasebank",
    ],
    "spam_toxicity": [
        "enron_spam",
        "sms_spam",
        "telegram-spam-ham",
        "hate_speech",
        "hate_speech_and_offensive_language",
        "jigsaw_toxicity_pred",
        "yelp_polarity",
    ],
    "machine_text_detection": [
        "corpus-of-diverse-styles",
        "gede_essay_detection",
        "machine_text_detection_DetectRL_direct_prompt",
        "machine_text_detection_M4_arxiv",
        "machine_text_detection_MAGE_tldr",
        "core",
    ],
}


def generate_scores(
    n_models: int,
    rng: np.random.Generator,
) -> dict:
    """Generate synthetic v_measure scores for all datasets.

    Each cluster group shares a latent "model quality" ranking, so datasets
    within the same group will have high Spearman correlation. Different groups
    use independent latent rankings.

    Args:
        n_models: Number of synthetic models to generate.
        rng: NumPy random generator for reproducibility.

    Returns:
        A dict mapping (model_name, dataset_name) -> score.
    """
    model_names = [f"model_{i:02d}" for i in range(n_models)]
    scores = {}

    for group_name, datasets in CLUSTER_GROUPS.items():
        # Each group has its own latent model ranking
        latent = rng.uniform(0.3, 0.9, size=n_models)

        for dataset in datasets:
            # Add per-dataset noise to the latent ranking (small enough to
            # preserve rank correlation within the group)
            noise = rng.normal(0, 0.03, size=n_models)
            raw = np.clip(latent + noise, 0.0, 1.0)

            for model_name, score in zip(model_names, raw):
                scores[(model_name, dataset)] = round(float(score), 4)

    return scores


def write_results(
    scores: dict,
    output_dir: str,
) -> None:
    """Write dummy metrics.json files to the output directory.

    Args:
        scores: Dict mapping (model_name, dataset_name) -> score.
        output_dir: Root directory for dummy results.
    """
    for (model_name, dataset_name), score in scores.items():
        path = os.path.join(
            output_dir,
            dataset_name,
            model_name,
            EPISODE_PARAMS,
            PATH_TASK,
        )
        os.makedirs(path, exist_ok=True)

        metrics_file = os.path.join(path, "metrics.json")
        with open(metrics_file, "w") as f:
            json.dump({METRIC: score}, f, indent=2)

    # Count what we wrote
    datasets = sorted({d for _, d in scores})
    models = sorted({m for m, _ in scores})
    print(f"Wrote {len(scores)} metrics files")
    print(f"  {len(models)} models: {models[0]} .. {models[-1]}")
    print(f"  {len(datasets)} datasets")
    print(f"  Output: {output_dir}")


def main() -> None:
    """Generate dummy results for testing benchmark clustering."""
    rng = np.random.default_rng(SEED)
    scores = generate_scores(N_MODELS, rng)
    write_results(scores, DUMMY_DIR)


if __name__ == "__main__":
    main()
