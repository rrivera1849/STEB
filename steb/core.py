import importlib
import json
import os
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from termcolor import colored
from tqdm import tqdm
from transformers import set_seed

from .dataset_loader import DatasetLoader
from .models import get_model_registry
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
    from transformers.models.auto.modeling_auto import (
        MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
        MODEL_FOR_MASKED_LM_MAPPING_NAMES,
    )

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
    from transformers import AutoConfig

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
    model_class = None
    registry = get_model_registry()
    # Allow models to be referenced with prefixes, e.g. "lftk:config.yaml" or
    # "tfidfngrams:/path/to/vectorizers.pkl" by matching on the part before ":".
    prefix = model_name_or_path.split(":", 1)[0]
    for model_cls in registry.values():
        if prefix in getattr(model_cls, "supported_models", []):
            model_class = model_cls
            break
    if model_class is None:
        model_class = registry["hf"]
    return model_class(model_name_or_path)


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


def evaluate(
    model,
    datasets: List[str],
    episode_sizes: List[int],
    task_name: Optional[str] = None,
    n_episodes_per_class: int = 50,
    batch_size: int = 32,
    force_reload: bool = False,
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
        episode_sizes: A list of episode sizes to evaluate.
        task_name: The name of the task to evaluate. If None, runs all tasks.
        n_episodes_per_class: The number of episodes per class.
        batch_size: The batch size for embedding.
        force_reload: Whether to force reload the datasets.
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

    def extract_features(dataset, episode_size, n_episodes_per_class, batch_size, show_progress=False):
        """
        Extracts features from the dataset using the specified model.

        Expects dataset format:
            Order Alignment: {"label": [[seq1_most, ..., seq1_least], [seq2_most, ..., seq2_least], ...]}
            Others: {"label": [[text_1, ..., text_N], [text_1, ..., text_M], ...]}

        Each label maps to a list of ordered sequences. Sequences are grouped into
        episodes, then organized by position (most X, ..., least X).
        """
        episodes_by_label = {}
        for label, text_list in dataset.items():
            # Validate nested list format
            assert text_list and isinstance(text_list[0], list), \
                f"Dataset for label '{label}' must be a list of lists (ordered sequences)"

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

    dataset_iterator = tqdm(datasets, desc="Evaluating Datasets", disable=not progress_bar)
    for dataset_name in dataset_iterator:
        for episode_size in episode_sizes:
            print(colored(f"--- Evaluating {dataset_name} (episode size: {episode_size}) ---", "cyan"))

            try:
                dset_loader = DatasetLoader(
                    dataset_name=dataset_name,
                    episode_size=episode_size,
                    n_episodes_per_class=n_episodes_per_class,
                    force_reload=force_reload,
                    seed=seed,
                )

                with open(dset_loader.config_path) as f:
                    config = json.load(f)

                tasks_to_run = [task_name] if task_name else list(config.get("tasks", {}).keys())

                # Only load the default (non-task-specific) dataset if at least
                # one task uses the top-level record handler.
                default_X, default_y = None, None
                needs_default = any(
                    "record_handler" not in config.get("tasks", {}).get(t, {})
                    for t in tasks_to_run
                )
                if needs_default:
                    dataset = dset_loader.load()
                    if not dataset:
                        continue
                    default_X, default_y = extract_features(
                        dataset, episode_size, n_episodes_per_class, batch_size, show_progress=progress_bar,
                    )
            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                print(colored(f"  FAILED to load dataset: {error_msg}", "red"))
                traceback.print_exc()
                failures.append((dataset_name, episode_size, "dataset_load", error_msg))
                continue

            for current_task_name in tasks_to_run:
                print(colored(f"  - Running task: {current_task_name}", "blue"))
                task_config = config.get("tasks", {}).get(current_task_name)
                if not task_config:
                    print(colored(f"Task '{current_task_name}' not supported by dataset '{dataset_name}'. Skipping.", "yellow"))
                    continue

                model_str = os.path.basename(model.model_name_or_path)
                if model_str == "":
                    model_str = os.path.basename(os.path.dirname(model.model_name_or_path))
                dset_str = os.path.basename(dataset_name)
                scores_path = os.path.join(
                    output_folder, dset_str, model_str,
                    f"{episode_size}_{n_episodes_per_class}", current_task_name,
                )
                metrics_path = os.path.join(scores_path, "metrics.json")

                if not force_reload and os.path.exists(metrics_path):
                    print(colored(f"    -> Skipping (results already exist)", "yellow"))
                    successes.append((dataset_name, episode_size, current_task_name))
                    continue

                try:
                    if "record_handler" in task_config:
                        task_loader = DatasetLoader(
                            dataset_name=dataset_name,
                            episode_size=episode_size,
                            n_episodes_per_class=n_episodes_per_class,
                            force_reload=force_reload,
                            seed=seed,
                            task_name=current_task_name,
                        )
                        task_dataset = task_loader.load()
                        if not task_dataset:
                            continue
                        current_X, current_y = extract_features(
                            task_dataset, episode_size, n_episodes_per_class, batch_size, show_progress=progress_bar,
                        )
                    else:
                        current_X, current_y = default_X, default_y

                    processor_module = importlib.import_module(f"steb.processors.{task_config['processor']}")
                    processor_class_name = f"{task_config['processor'].replace('_', ' ').title().replace(' ', '')}Processor"
                    processor_class = getattr(processor_module, processor_class_name)
                    processor = processor_class()

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
