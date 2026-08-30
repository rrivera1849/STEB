"""Configuration constants for benchmark_clustering."""
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple


# Maps task name -> primary metric
TASK_METRICS: Dict[str, str] = {
    "clustering": "v_measure",
    "all_to_all_pair_classification": "auc",
    "pre_defined_pair_classification": "auc",
    "order_alignment": "distractor_acc_mean",
    "retrieval": "mrr",
    "probing": "average",
}

# Recognised --oa_variant values for cluster YAML entries.
# Each maps the variant name to the metric used for the order_alignment task
# when that variant is selected. --oa_variant only controls the metric; use
# --oa_only to additionally restrict an entry to the order_alignment task.
OA_VARIANT_METRICS: Dict[str, str] = {
    "distractor": "distractor_acc_mean",
    "acc": "acc_mean",
}


@dataclass(frozen=True)
class ClusterEntry:
    """A parsed cluster YAML entry.

    Attributes:
        name: The dataset name.
        oa_variant: Which order_alignment metric to use for this entry —
            None (use TASK_METRICS default), "distractor", or "acc".
        oa_only: If True, only the order_alignment task contributes to the
            cluster table for this entry; other tasks the dataset declares
            are dropped.
    """
    name: str
    oa_variant: Optional[Literal["distractor", "acc"]] = None
    oa_only: bool = False

# Datasets where the label is primarily semantic (topic, sentiment, content)
# rather than stylistic. Excluded from analysis by default.
SEMANTIC_DATASETS = {
    # Topic / content-based
    "20_Newsgroups_Fixed",
    "ag_news",
    "reuters21578",
    # Sentiment / emotion
    "emotion",
    "financial_phrasebank",
    "twitter-airline-sentiment",
    "yelp_polarity",
    # MISC (not semantic, but I don't want it)
    "probing_blog_small",
    "dummy_retrieval",
    "dummy_order_alignment",
    "fast_baseline_compare_234grams",
    "lftk_sweep",
    "lftk_sweep_fast",
    "lftk_sweep_fast_surfaceavg",
    "lftk_sweep_fast_surfacepos",
}

# Non-English datasets. Excluded from analysis by default.
NON_ENGLISH_DATASETS = {
    # PAN13
    "pan13_authorship_verification_greek_test",
    "pan13_authorship_verification_spanish_test",
    # PAN14
    "pan14_authorship_verification_corpus1_dutch_essays_test",
    "pan14_authorship_verification_corpus1_dutch_reviews_test",
    "pan14_authorship_verification_corpus1_greek_articles_test",
    "pan14_authorship_verification_corpus1_spanish_articles_test",
    "pan14_authorship_verification_corpus2_dutch_essays_test",
    "pan14_authorship_verification_corpus2_dutch_reviews_test",
    "pan14_authorship_verification_corpus2_greek_articles_test",
    "pan14_authorship_verification_corpus2_spanish_articles_test",
    # PAN15
    "pan15_authorship_verification_dutch_test",
    "pan15_authorship_verification_greek_test",
    "pan15_authorship_verification_spanish_test",
    # PAN18
    "pan18_cross_domain_authorship_attribution_french",
    "pan18_cross_domain_authorship_attribution_italian",
    "pan18_cross_domain_authorship_attribution_polish",
    "pan18_cross_domain_authorship_attribution_spanish",
}

EXCLUDED_DATASETS = SEMANTIC_DATASETS | NON_ENGLISH_DATASETS

# Models to exclude from analysis (e.g. broken runs, non-comparable baselines).
EXCLUDED_MODELS: set[str] = set()
EXCLUDED_MODELS.add("avgs_typetoken_read.yaml")
EXCLUDED_MODELS.add("lftk")
EXCLUDED_MODELS.add("tfidf")
EXCLUDED_MODELS.add("tfidfngrams")

LOW_CONFIDENCE_THRESHOLD = 10

# Aggregate sheets: each entry defines a new Excel sheet that concatenates
# columns from multiple manual cluster tables side by side.
# Format: (sheet_name, [avg_cluster, ...]) or
#         (sheet_name, [avg_cluster, ...], [extra_cluster, ...])
# Clusters listed in avg contribute to the `average` column; extras are
# shown in the sheet but excluded from that average.
AGGREGATE_SHEETS: List[Tuple] = [
    ("summary_tasks",
     ["machine_text_detection", "machine_text_detection_adversarial", "authorship_verification", "authorship_retrieval"],
     ["authorship_verification_easy", "authorship_verification_medium", "authorship_verification_hard"]),
    ("summary_features", ["genre", "register", "style_vs_content", "time", "demographics", "feature_probing", "predefined_features", "dialect"]),
]

# Definitional STEB score (Wegmann et al. 2026, cf. paper §3.3 + appendix
# `tab:dataset-style-score` / `tab:attribute-clusters`). Mirrors the paper's
# table column-for-column: the six Object of Study sub-clusters as their own
# columns, then their average, then Linguistic Features, Content Independence,
# and the final Definitional score (average of the three axis-level columns).
#
# Each entry is ``(column_name, kind, payload)`` where ``kind`` is:
#   "cluster"  – column = mean over the named manual cluster's datasets
#   "subaxes"  – column = mean of the listed columns already computed above
DEFINITIONAL_SHEET_NAME = "STEB_definitional"
DEFINITIONAL_COLUMNS: List[Tuple[str, str, object]] = [
    ("Genre", "cluster", "genre"),
    ("Register", "cluster", "register"),
    ("Time", "cluster", "time"),
    ("Demographics", "cluster", "demographics"),
    ("Dialect", "cluster", "dialect"),
    ("Idiolect", "cluster", "idiolect"),
    ("Object of Study", "subaxes",
     ["Genre", "Register", "Time", "Demographics", "Dialect", "Idiolect"]),
    ("Linguistic Features", "cluster", "predefined_features"),
    ("Content Independence", "cluster", "style_vs_content"),
    ("Definitional score", "subaxes",
     ["Object of Study", "Linguistic Features", "Content Independence"]),
]

# Model categories for ranking plot. Models not listed get "other" styling.
MODEL_CATEGORIES: Dict[str, List[str]] = {
    "style": [
        "LUAR-CRUD",
        "LUAR-MUD",
        "Style-Embedding",
        "star",
        "styledistance",
        "lisa_checkpoint",
        "LUSR",
    ],
    "multilingual": [
        "mstyledistance",
        "multilingual-style-representation",
        "multilingual-style-representation-Llama-3.2",
    ],
    "semantic": [
        "Qwen3-Embedding-8B",
        "all-mpnet-base-v2",
        "bge-base-en-v1.5",
        "bge-large-en-v1.5",
        "e5-base-v2",
        "e5-large-v2",
        "e5-mistral-7b-instruct",
        "gte-base-en-v1.5",
        "gte-large-en-v1.5",
        "jina-embeddings-v3",
    ],
}
