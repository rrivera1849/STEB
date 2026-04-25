import importlib
import json
import os
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from termcolor import colored
from tqdm import tqdm
from transformers import AutoConfig, set_seed
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_MASKED_LM_MAPPING_NAMES,
)

from .dataset_loader import DatasetLoader
from .models import MODEL_REGISTRY
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
    "clustering": {"episode_sizes": [1], "n_episodes_per_class": 50},
    "all_to_all_pair_classification": {"episode_sizes": [1], "n_episodes_per_class": 50},
    "order_alignment": {"episode_sizes": [1], "n_episodes_per_class": 50},
    "pre_defined_pair_classification": {"episode_sizes": [1], "n_episodes_per_class": 2},
    "probing": {"episode_sizes": [1], "n_episodes_per_class": 1},
    "retrieval": {"episode_sizes": [1], "n_episodes_per_class": 1},
}

# Tasks for which auto_submetric_per_label defaults to true. A dataset can
# still override per-task via "auto_submetric_per_label": false in its config.
# Validation rejects the flag on tasks not in this set.
AUTO_PER_LABEL_TASKS = {"order_alignment"}

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
        True if the model is a causal LM, False otherwise.
    """
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
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


def get_model(model_name_or_path: str):
    """
    Loads a STEB model.

    Checks the supported_models lists first, then auto-detects whether the
    model is a causal LM or an encoder model, and loads accordingly.

    Args:
        model_name_or_path: The name or path of the model to load.

    Returns:
        An instance of a STEBModel.
    """
    for model_cls in MODEL_REGISTRY.values():
        if model_name_or_path in model_cls.supported_models:
            return model_cls(model_name_or_path)

    if _is_causal_model(model_name_or_path):
        return MODEL_REGISTRY["causal"](model_name_or_path)

    return MODEL_REGISTRY["hf"](model_name_or_path)


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

        # RRS - Removing for now, but we want something more intelligent here.
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


def evaluate(
    model,
    datasets: List[str],
    episode_sizes: Optional[List[int]] = None,
    task_name: Optional[str] = None,
    n_episodes_per_class: Optional[int] = None,
    batch_size: int = 32,
    force_reload: bool = False,
    force_rerun: bool = False,
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

    package_dir = os.path.dirname(os.path.abspath(__file__))

    dataset_iterator = tqdm(datasets, desc="Evaluating Datasets", disable=not progress_bar)
    for dataset_name in dataset_iterator:
        config_path = os.path.join(package_dir, "steb_datasets", dataset_name, "config.json")
        try:
            with open(config_path) as f:
                config = json.load(f)
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            print(colored(f"  FAILED to read config for {dataset_name}: {error_msg}", "red"))
            traceback.print_exc()
            failures.append((dataset_name, -1, "config_read", error_msg))
            continue

        tasks_to_run = [task_name] if task_name else list(config.get("tasks", {}).keys())

        # Cache default embeddings by (episode_size, n_episodes_per_class)
        # so tasks sharing the same parameters reuse the same embeddings.
        default_cache: Dict[Tuple[int, int], Tuple[Any, Any]] = {}

        for current_task_name in tasks_to_run:
            task_config = config.get("tasks", {}).get(current_task_name)
            if task_config is None:
                print(colored(f"Task '{current_task_name}' not supported by dataset '{dataset_name}'. Skipping.", "yellow"))
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
                print(colored(f"--- Evaluating {dataset_name} | {current_task_name} (episode size: {episode_size}) ---", "cyan"))

                model_str = os.path.basename(model.model_name_or_path)
                if model_str == "":
                    model_str = os.path.basename(os.path.dirname(model.model_name_or_path))
                dset_str = os.path.basename(dataset_name)
                scores_path = os.path.join(
                    output_folder, dset_str, model_str,
                    f"{episode_size}_{resolved_n_episodes}", current_task_name,
                )
                metrics_path = os.path.join(scores_path, "metrics.json")

                if not force_rerun and os.path.exists(metrics_path):
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
                        current_X, current_y = extract_features(
                            task_dataset, episode_size, resolved_n_episodes, batch_size, show_progress=progress_bar,
                        )
                    else:
                        cache_key = (episode_size, resolved_n_episodes)
                        if cache_key not in default_cache:
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
                            default_cache[cache_key] = extract_features(
                                dataset, episode_size, resolved_n_episodes, batch_size, show_progress=progress_bar,
                            )
                        current_X, current_y = default_cache[cache_key]

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

                    metrics = task.evaluate(*processed_data)

                    # Tasks emit per-label results under the internal key
                    # "_per_label" for order_alignment; other tasks default
                    # to off (and validation rejects the flag on them currently).
                    # Either way, strip the internal key before serialisation
                    # so it never appears at the top level.
                    internal_per_label = metrics.pop("_per_label", None)
                    auto_per_label = task_config.get(
                        "auto_submetric_per_label",
                        current_task_name in AUTO_PER_LABEL_TASKS,
                    )

                    submetrics_config = task_config.get("submetrics", {})
                    submetrics_out: Dict[str, Any] = {}
                    if auto_per_label and internal_per_label is not None:
                        submetrics_out.update(internal_per_label)
                    if submetrics_config:
                        submetrics_out.update(_evaluate_submetrics(
                            submetrics_config,
                            processed_data,
                            task,
                        ))
                    if submetrics_out:
                        metrics["submetrics"] = submetrics_out

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
