import importlib
import json
import os
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from termcolor import colored
from transformers import AutoConfig, set_seed
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_MASKED_LM_MAPPING_NAMES,
)

from .dataset_loader import DatasetLoader
from .models import get_model_registry
from .models.lisa_model import is_lisa_model
from .models.sentence_transformer_model import is_sentence_transformer_model
from .processors.base import Processor
from .steb_datasets import DATASET_REGISTRY
from .utils import RESULTS_DIR

SUPPORTED_TASKS = [
    "all_to_all_pair_classification",
    "clustering",
    "order_alignment",
    "pre_defined_pair_classification",
    "retrieval",
    "probing",
]

TASK_DEFAULTS = {
    "clustering": {"episode_sizes": [1], "n_episodes_per_class": "auto"},
    "all_to_all_pair_classification": {"episode_sizes": [1], "n_episodes_per_class": "auto"},
    "order_alignment": {"episode_sizes": [1], "n_episodes_per_class": "auto"},
    "pre_defined_pair_classification": {"episode_sizes": [1], "n_episodes_per_class": 2},
    "probing": {"episode_sizes": [1], "n_episodes_per_class": 1},
    "retrieval": {"episode_sizes": [-1], "n_episodes_per_class": 1},
}

def get_supported_tasks() -> list[str]:
    """Returns the list of all supported task names."""
    return SUPPORTED_TASKS


def _get_causal_only_model_types() -> set:
    """
    Returns the set of model types that are exclusively causal LMs.

    Computes the difference between HuggingFace's causal LM mapping and
    masked LM mapping. Models that appear in both (e.g. BERT, RoBERTa) are
    primarily encoders that happen to have a causal variant, so they are
    excluded.

    Returns:
        A frozenset of model_type strings for causal-only architectures.
    """
    causal_types = set(MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.keys())
    masked_types = set(MODEL_FOR_MASKED_LM_MAPPING_NAMES.keys())
    return causal_types - masked_types


def _is_causal_model(model_name_or_path: str) -> bool:
    """
    Checks whether a model is a causal (auto-regressive) language model.

    Detection strategy (in order):
    1. If the config's architectures list contains a class name with
       "ForCausalLM" or "LMHeadModel", and the model_type is NOT a
       primarily-encoder type (e.g. BERT, RoBERTa), return True.
    2. If the model_type is in the causal-only set (derived from HuggingFace's
       MODEL_FOR_CAUSAL_LM_MAPPING minus MODEL_FOR_MASKED_LM_MAPPING), return True.

    Args:
        model_name_or_path: The name or path of the model.

    Returns:
        True if the model is a causal LM, False otherwise. Models without a
        loadable HuggingFace config (e.g. adapter-only repos) return False
        instead of raising.
    """
    try:
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    except (OSError, ValueError):
        return False
    causal_only_types = _get_causal_only_model_types()

    model_type = getattr(config, "model_type", "")
    if model_type in causal_only_types:
        return True

    # Check architecture class names for models not in the standard mappings
    architectures = getattr(config, "architectures", []) or []
    causal_arch_patterns = ("ForCausalLM", "LMHeadModel")
    for arch in architectures:
        if any(pattern in arch for pattern in causal_arch_patterns):
            return True

    return False


def get_model(
    model_name_or_path: str,
    truncate: bool = False,
    max_tokens: Optional[int] = None,
):
    """
    Loads a STEB model.

    Dispatches in four stages:
      1. Match the prefix of ``model_name_or_path`` (the part before ``":"``)
         against each registered class's ``supported_models`` list.
      2. If the path points at a LISA checkpoint directory (detected via
         :func:`is_lisa_model`), route to :class:`LISAModel`.
      3. If the checkpoint is in the sentence-transformers-only format
         (``modules.json`` without a top-level ``config.json``, detected via
         :func:`is_sentence_transformer_model`), route to
         :class:`SentenceTransformerModel`.
      4. If nothing matched, inspect the HuggingFace config and route
         auto-regressive LMs to :class:`CausalModel`.
      5. Otherwise fall back to :class:`HFModel`.

    Args:
        model_name_or_path: The name or path of the model to load.
        truncate: If True, truncate each text to the token cap instead of
            chunking and mean-pooling. No-op for non-tokenizer models.
        max_tokens: Optional per-text token cap, capped at the model's
            native maximum. No-op for non-tokenizer models.

    Returns:
        An instance of a STEBModel.
    """
    kwargs = {"truncate": truncate, "max_tokens": max_tokens}

    registry = get_model_registry()
    # Allow models to be referenced with prefixes, e.g. "lftk:config.yaml" or
    # "tfidfngrams:/path/to/vectorizers.pkl" by matching on the part before ":".
    prefix = model_name_or_path.split(":", 1)[0]
    for model_cls in registry.values():
        if prefix in getattr(model_cls, "supported_models", []):
            return model_cls(model_name_or_path, **kwargs)

    if is_lisa_model(model_name_or_path):
        return registry["lisa"](model_name_or_path, **kwargs)

    if is_sentence_transformer_model(model_name_or_path):
        return registry["sentence_transformer"](model_name_or_path, **kwargs)

    if _is_causal_model(model_name_or_path):
        return registry["causal"](model_name_or_path, **kwargs)

    return registry["hf"](model_name_or_path, **kwargs)


def get_all_datasets() -> List[str]:
    """
    Retrieves a list of all available STEB datasets.

    Returns:
        A list of all available dataset names.
    """
    return DATASET_REGISTRY


def get_supported_datasets(task_name: str) -> List[str]:
    """
    Retrieves a list of datasets that support the given task.

    Args:
        task_name: The name of the task.

    Returns:
        A list of supported dataset names.
    """
    supported_datasets = []
    package_dir = os.path.dirname(os.path.abspath(__file__))
    for dataset_name in DATASET_REGISTRY:
        config_path = os.path.join(package_dir, "steb_datasets", dataset_name, "config.json")
        with open(config_path) as f:
            config = json.load(f)
        if task_name in config.get("tasks", {}):
            supported_datasets.append(dataset_name)
    return supported_datasets


def _print_evaluation_summary(
    successes: List[Tuple[str, int, str]],
    failures: List[Tuple[str, int, str, str]],
) -> None:
    """
    Prints a summary of evaluation results.

    Args:
        successes: List of (dataset, episode_size, task) tuples that succeeded.
        failures: List of (dataset, episode_size, task, error_message) tuples that failed.
    """
    print()
    print(colored("=" * 60, "cyan"))
    print(colored("Evaluation Summary", "cyan"))
    print(colored("=" * 60, "cyan"))
    print(colored(f"  Succeeded: {len(successes)}", "green"))
    print(colored(f"  Failed:    {len(failures)}", "red" if failures else "green"))

    if failures:
        print()
        print(colored("Failures:", "red"))
        for dataset, episode_size, task, error_msg in failures:
            print(colored(f"  - {dataset} | episode_size={episode_size} | {task}", "red"))
            print(colored(f"    {error_msg}", "red"))

    print(colored("=" * 60, "cyan"))


def _write_evaluation_log(
    output_folder: str,
    model_name: str,
    run_name: Optional[str],
    successes: List[Tuple[str, int, str]],
    failures: List[Tuple[str, int, str, str]],
) -> str:
    """
    Writes a JSON log of the evaluation run to the output folder.

    Args:
        output_folder: Directory to write the log file into.
        model_name: Model identifier used for the run.
        run_name: Optional run identifier (e.g. preset name or task).
        successes: List of (dataset, episode_size, task) tuples that succeeded.
        failures: List of (dataset, episode_size, task, error_message) tuples that failed.

    Returns:
        The path to the written log file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [model_name]
    if run_name:
        parts.append(run_name)
    parts.append(timestamp)
    filename = "_".join(parts) + ".log.json"

    log: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "run_name": run_name,
        "num_succeeded": len(successes),
        "num_failed": len(failures),
        "successes": [
            {"dataset": d, "episode_size": e, "task": t}
            for d, e, t in successes
        ],
        "failures": [
            {"dataset": d, "episode_size": e, "task": t, "error": msg}
            for d, e, t, msg in failures
        ],
    }

    logs_dir = os.path.join(output_folder, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, filename)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(colored(f"  Log written to: {log_path}", "cyan"))
    return log_path


def _evaluate_submetrics(
    submetrics_config: Dict[str, List[str]],
    processed_data: Any,
    task: Any,
) -> Dict[str, Any]:
    """
    Evaluates submetrics by filtering processed data to label subsets.

    Args:
        submetrics_config: Mapping of submetric name to list of labels to keep.
        processed_data: Tuple of (X, y) from the processor.
        task: The task instance to call evaluate() on.

    Returns:
        A dict mapping submetric names to their metric dicts.
    """
    all_X, all_y = processed_data
    submetrics = {}

    for sub_name, label_subset in submetrics_config.items():
        label_set = set(label_subset)
        filtered = [
            (x, y) for x, y in zip(all_X, all_y)
            if y in label_set
        ]

        # I think Order Alignment is the only task where you can have one label.
        # unique_labels = set(y for _, y in filtered)
        # if len(unique_labels) < 2:
        #     msg = f"only {len(unique_labels)} unique label(s) found, need at least 2"
        #     print(colored(f"    FAILED submetric '{sub_name}': {msg}", "red"))
        #     submetrics[sub_name] = {"error": msg}
        #     continue

        try:
            sub_X, sub_y = zip(*filtered)
            sub_metrics = task.evaluate(list(sub_X), list(sub_y))
            submetrics[sub_name] = sub_metrics
            print(colored(f"    -> Submetric '{sub_name}': {sub_metrics}", "green"))
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            print(colored(f"    FAILED submetric '{sub_name}': {error_msg}", "red"))
            traceback.print_exc()
            submetrics[sub_name] = {"error": error_msg}

    return submetrics


def _iter_task_configs(
    datasets: List[str],
    task_name: Optional[str] = None,
    episode_sizes: Optional[List[int]] = None,
    n_episodes_per_class: Optional[int] = None,
):
    """
    Iterates over dataset/task/episode_size combinations, resolving defaults.

    Yields:
        Tuples of (dataset_name, current_task_name, task_config, episode_size,
        resolved_n_episodes) for each valid combination.
    """
    package_dir = os.path.dirname(os.path.abspath(__file__))

    for dataset_name in datasets:
        config_path = os.path.join(package_dir, "steb_datasets", dataset_name, "config.json")
        try:
            with open(config_path) as f:
                config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            yield (dataset_name, None, str(e), None, None)
            continue

        tasks_to_run = [task_name] if task_name else list(config.get("tasks", {}).keys())

        for current_task_name in tasks_to_run:
            task_config = config.get("tasks", {}).get(current_task_name)
            if task_config is None:
                continue

            task_defaults = TASK_DEFAULTS.get(current_task_name, {})
            resolved_episode_sizes = (
                episode_sizes
                or task_config.get("episode_sizes")
                or task_defaults.get("episode_sizes")
            )
            resolved_n_episodes = (
                n_episodes_per_class
                or task_config.get("n_episodes_per_class")
                or task_defaults.get("n_episodes_per_class")
            )

            for episode_size in resolved_episode_sizes:
                yield (dataset_name, current_task_name, task_config, episode_size, resolved_n_episodes)


def preview(
    datasets: List[str],
    task_name: Optional[str] = None,
    episode_sizes: Optional[List[int]] = None,
    n_episodes_per_class: Optional[int] = None,
    output_file: Optional[str] = None,
    show_summary: bool = True,
) -> List[Dict[str, Any]]:
    """
    Previews dataset statistics for a benchmark run without loading a model.

    For each dataset/task/episode_size combination, reports class counts,
    resolved n_episodes_per_class, and which classes would be dropped.

    Args:
        datasets: A list of dataset names to preview.
        task_name: The task to preview. If None, previews all tasks.
        episode_sizes: Episode sizes to preview. If None, uses per-task defaults.
        n_episodes_per_class: Override for n_episodes_per_class. If None, uses per-task defaults.
        output_file: Optional path to write the preview report to.
        show_summary: Whether to print the summary block. Set to False when
            the caller handles its own summary (e.g. preset mode).

    Returns:
        A list of result dicts, one per dataset/task/episode_size combination.
    """
    results = []
    lines = []

    for dataset_name, current_task_name, task_config, episode_size, resolved_n_episodes in _iter_task_configs(
        datasets, task_name, episode_sizes, n_episodes_per_class,
    ):
        if current_task_name is None:
            error_msg = task_config  # sentinel: error string stored in task_config slot
            print(colored(f"  {dataset_name} | ERROR: {error_msg}", "red"))
            continue

        try:
            loader = DatasetLoader(
                dataset_name=dataset_name,
                episode_size=episode_size,
                n_episodes_per_class=resolved_n_episodes,
                task_name=current_task_name,
            )
            stats = loader.preview()

            result = {
                "dataset": dataset_name,
                "task": current_task_name,
                **stats,
            }
            results.append(result)

            dropped_count = stats["dropped_classes"]
            status_str = "OK" if dropped_count == 0 else f"{dropped_count} dropped"
            plain_line = (
                f"  {dataset_name} | {current_task_name} | "
                f"ep_size={episode_size} | "
                f"n_episodes={stats['n_episodes_per_class']} | "
                f"classes={stats['kept_classes']}/{stats['total_classes']} | "
                f"min_count={stats['min_class_count']} | "
                f"{status_str}"
            )
            color = "green" if dropped_count == 0 else "yellow"
            print(colored(plain_line, color))
            lines.append(plain_line)

            if stats["dropped_labels"]:
                for label, count in sorted(stats["dropped_labels"].items(), key=lambda x: x[1]):
                    drop_line = f"    dropped '{label}': {count}/{stats['samples_per_class']} samples"
                    print(colored(drop_line, "yellow"))
                    lines.append(drop_line)

        except Exception as e:
            msg = f"  FAILED {dataset_name}/{current_task_name}: {type(e).__name__}: {e}"
            print(colored(msg, "red"))
            lines.append(msg)

    # Summary
    total_combos = len(results)
    total_dropped = sum(r["dropped_classes"] for r in results)
    total_kept = sum(r["kept_classes"] for r in results)
    total_total = sum(r["total_classes"] for r in results)

    if show_summary:
        summary_lines = [
            "",
            "=" * 60,
            "Preview Summary",
            "=" * 60,
            f"  Combinations: {total_combos}",
            f"  Classes kept: {total_kept}/{total_total}",
            f"  Classes dropped: {total_dropped}",
            "=" * 60,
        ]
        for line in summary_lines:
            print(colored(line, "cyan"))
        lines.extend(summary_lines)

    if output_file:
        with open(output_file, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(colored(f"  Report saved to: {output_file}", "cyan"))

    return results


def evaluate(
    model,
    datasets: List[str],
    episode_sizes: Optional[List[int]] = None,
    task_name: Optional[str] = None,
    n_episodes_per_class: Optional[int] = None,
    batch_size: int = 32,
    force_reload: bool = False,
    force_rerun: bool = False,
    force_rerun_oa: bool = False,
    progress_bar: bool = False,
    output_folder: str = RESULTS_DIR,
    seed: int = 42,
    run_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluates a model on a list of datasets for a given task.

    Individual dataset/task failures are caught and logged rather than
    crashing the entire run. A summary of successes and failures is
    printed at the end, and a JSON log is written to
    ``{output_folder}/logs/``.

    Args:
        model: The model to evaluate.
        datasets: A list of dataset names to evaluate on.
        episode_sizes: A list of episode sizes to evaluate. If None,
            uses per-task defaults from TASK_DEFAULTS.
        task_name: The name of the task to evaluate. If None, runs all tasks.
        n_episodes_per_class: The number of episodes per class. If None,
            uses per-task defaults from TASK_DEFAULTS.
        batch_size: The batch size for embedding.
        force_reload: Whether to force reload the datasets.
        force_rerun: Whether to re-run evaluations even if metrics already exist.
        force_rerun_oa: Whether to re-run the order_alignment task only,
            even if its metrics file already exists. Ignored when
            ``force_rerun`` is also set.
        progress_bar: Whether to show a progress bar.
        output_folder: The folder to save the results to.
        seed: The random seed to use.
        run_name: An optional label for this run (e.g. preset name).
            Used in the log filename alongside the model name.

    Returns:
        A dictionary with "successes", "failures", and "log_path" keys.
    """
    set_seed(seed)

    successes: List[Tuple[str, int, str]] = []
    failures: List[Tuple[str, int, str, str]] = []

    def safe_load(
        loader: DatasetLoader,
        dataset_name: str,
        episode_size: int,
        current_task_name: str,
    ) -> Optional[Dict]:
        """
        Attempts to load a dataset, logging and recording the failure if it occurs.

        Args:
            loader: The DatasetLoader to call load() on.
            dataset_name: Name of the dataset (for error reporting).
            episode_size: Current episode size (for error reporting).
            current_task_name: Current task name (for error reporting).

        Returns:
            The loaded dataset, or None if loading failed.
        """
        try:
            return loader.load()
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            print(colored(f"    FAILED to load dataset: {error_msg}", "red"))
            traceback.print_exc()
            failures.append((dataset_name, episode_size, current_task_name, error_msg))
            return None

    def extract_features(
        dataset,
        episode_size,
        n_episodes_per_class,
        batch_size,
        show_progress=False,
    ):
        """
        Extracts features from the dataset using the specified model.

        Expects dataset format:
            Order Alignment: {"label": [[seq1_most, ..., seq1_least], [seq2_most, ..., seq2_least], ...]}
                Each label maps to a list of ordered sequences. Sequences are grouped into
                episodes, then organized by position (most X, ..., least X).
            Others: {"label": [[text_1, ..., text_N], [text_1, ..., text_M], ...]}

        """
        episodes_by_label = {}
        for label, text_list in dataset.items():
            # Validate nested list format
            assert text_list and isinstance(text_list[0], list), \
                f"Dataset for label '{label}' must be a list of lists"

            seq_len = len(text_list[0])
            if episode_size == -1:
                if current_task_name == "pre_defined_pair_classification":
                    # Keep each text list as its own episode so pairs remain separate
                    episodes_by_label[label] = [[lst] for lst in text_list]
                else:
                    # Group all sequences into a single large episode
                    episodes_by_label[label] = [[[sublist for lst in text_list for sublist in lst]]]
            else:
                # Group sequences into episodes, organize by position
                episodes_by_label[label] = [
                    [[seq[pos] for seq in text_list[i:i+episode_size]] for pos in range(seq_len)]
                    for i in range(0, len(text_list), episode_size)
                ]
                assert len(episodes_by_label[label]) == n_episodes_per_class
                assert all(len(episode[0]) == episode_size for episode in episodes_by_label[label])
        all_episodes = [episode for label, episodes in episodes_by_label.items() for episode in episodes]
        y = [label for label, episodes in episodes_by_label.items() for _ in episodes]
        num_positions = len(all_episodes[0])
        if task_name == "order_alignment":
            assert all(len(episode) == num_positions for episode in all_episodes), \
                ("All entries must have the same number of positions, "
                "functionality for variable-length text sets not implemented.")

        # Flatten episodes for embedding: [[[pos0s], [pos1s], ...], ...] -> [[pos0s], [pos1s], [pos0s], [pos1s], ...]
        flat_episodes = [position for episode in all_episodes for position in episode]
        X_flat = model.embed_multiple(flat_episodes, batch_size, show_progress=show_progress)
        X = [X_flat[i:i+num_positions] for i in range(0, len(X_flat), num_positions)]
        return X, y

    # Cache default embeddings by (dataset, episode_size, n_episodes_per_class)
    # so tasks sharing the same parameters reuse the same embeddings.
    default_cache: Dict[Tuple[str, int, int], Tuple[Any, Any]] = {}

    for dataset_name, current_task_name, task_config, episode_size, resolved_n_episodes in _iter_task_configs(
        datasets, task_name, episode_sizes, n_episodes_per_class,
    ):
        if current_task_name is None:
            error_msg = task_config  # sentinel: error string stored in task_config slot
            print(colored(f"--- Skipping {dataset_name}: {error_msg} ---", "red"))
            failures.append((dataset_name, -1, "config", error_msg))
            continue

        print(colored(f"--- Evaluating {dataset_name} | {current_task_name} (episode size: {episode_size}) ---", "cyan"))

        model_str = os.path.basename(model.model_name_or_path)
        if model_str == "":
            model_str = os.path.basename(os.path.dirname(model.model_name_or_path))
        dset_str = os.path.basename(dataset_name)

        tokens_suffix = (
            f"tokens_{model.effective_max_tokens}"
            if getattr(model, "effective_max_tokens", None) is not None
            else None
        )

        # When n_episodes is not "auto", we can check for existing results early
        if resolved_n_episodes != "auto":
            scores_path = os.path.join(
                output_folder, dset_str, model_str,
                f"{episode_size}_{resolved_n_episodes}", current_task_name,
            )
            if tokens_suffix is not None:
                scores_path = os.path.join(scores_path, tokens_suffix)
            metrics_path = os.path.join(scores_path, "metrics.json")
            if (
                not force_rerun
                and not (force_rerun_oa and current_task_name == "order_alignment")
                and os.path.exists(metrics_path)
            ):
                print(colored(f"    -> Skipping (results already exist)", "yellow"))
                successes.append((dataset_name, episode_size, current_task_name))
                continue

        try:
            if "record_handler" in task_config:
                task_loader = DatasetLoader(
                    dataset_name=dataset_name,
                    episode_size=episode_size,
                    n_episodes_per_class=resolved_n_episodes,
                    force_reload=force_reload,
                    seed=seed,
                    task_name=current_task_name,
                )
                task_dataset = safe_load(task_loader, dataset_name, episode_size, current_task_name)
                if task_dataset is None:
                    continue
                actual_n_episodes = task_loader.n_episodes_per_class
                current_X, current_y = extract_features(
                    task_dataset, episode_size, actual_n_episodes, batch_size, show_progress=progress_bar,
                )
            else:
                dset_loader = DatasetLoader(
                    dataset_name=dataset_name,
                    episode_size=episode_size,
                    n_episodes_per_class=resolved_n_episodes,
                    force_reload=force_reload,
                    seed=seed,
                )
                dataset = safe_load(dset_loader, dataset_name, episode_size, current_task_name)
                if dataset is None:
                    continue
                actual_n_episodes = dset_loader.n_episodes_per_class
                cache_key = (dataset_name, episode_size, actual_n_episodes)
                if cache_key not in default_cache:
                    default_cache[cache_key] = extract_features(
                        dataset, episode_size, actual_n_episodes, batch_size, show_progress=progress_bar,
                    )
                current_X, current_y = default_cache[cache_key]

            # Resolve scores_path now that actual_n_episodes is known
            scores_path = os.path.join(
                output_folder, dset_str, model_str,
                f"{episode_size}_{actual_n_episodes}", current_task_name,
            )
            if tokens_suffix is not None:
                scores_path = os.path.join(scores_path, tokens_suffix)
            metrics_path = os.path.join(scores_path, "metrics.json")

            # Check for existing results after resolving "auto"
            if (
                resolved_n_episodes == "auto"
                and not force_rerun
                and not (force_rerun_oa and current_task_name == "order_alignment")
                and os.path.exists(metrics_path)
            ):
                print(colored(f"    -> Skipping (results already exist)", "yellow"))
                successes.append((dataset_name, episode_size, current_task_name))
                continue

            if "processor" in task_config:
                processor_module = importlib.import_module(f"steb.processors.{task_config['processor']}")
                processor_class_name = f"{task_config['processor'].replace('_', ' ').title().replace(' ', '')}Processor"
                processor_class = getattr(processor_module, processor_class_name)
                processor = processor_class()
            else:
                processor = Processor()

            processed_data = processor.process(current_X, current_y)

            task_module = importlib.import_module(f"steb.tasks.{current_task_name}")
            task_class_name = f"{current_task_name.replace('_', ' ').title().replace(' ', '')}Task"
            task_class = getattr(task_module, task_class_name)
            task = task_class()

            # LFTK uses abs-diff / L1-diff for pair tasks and clustering; others use cosine / K-Means
            from .models.lftk_model import LFTKModel

            is_lftk = isinstance(model, LFTKModel)
            if current_task_name in ("pre_defined_pair_classification", "all_to_all_pair_classification"):
                score_mode = "abs_diff" if is_lftk else "cosine"
                metrics = task.evaluate(*processed_data, score_mode=score_mode)
            elif current_task_name == "clustering":
                distance_mode = "l1_diff" if is_lftk else "euclidean"
                metrics = task.evaluate(*processed_data, distance_mode=distance_mode)
            else:
                metrics = task.evaluate(*processed_data)

            submetrics_config = task_config.get("submetrics", {})
            if submetrics_config:
                metrics["submetrics"] = _evaluate_submetrics(
                    submetrics_config,
                    processed_data,
                    task,
                )

            os.makedirs(scores_path, exist_ok=True)
            with open(metrics_path, "w+") as ouf:
                ouf.write(json.dumps(metrics))

            print(colored(f"    -> Metrics: {metrics}", "green"))
            successes.append((dataset_name, episode_size, current_task_name))

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            print(colored(f"  FAILED {dataset_name}/{current_task_name}: {error_msg}", "red"))
            traceback.print_exc()
            failures.append((dataset_name, episode_size, current_task_name, error_msg))

    _print_evaluation_summary(successes, failures)

    model_str = os.path.basename(model.model_name_or_path)
    if model_str == "":
        model_str = os.path.basename(os.path.dirname(model.model_name_or_path))

    log_path = _write_evaluation_log(
        output_folder,
        model_str,
        run_name,
        successes,
        failures,
    )
    return {"successes": successes, "failures": failures, "log_path": log_path}
