import argparse
import json
import os
import sys

from termcolor import colored

from steb import evaluate, get_all_datasets, get_model, get_supported_datasets
from steb.presets import PRESETS, resolve_preset
from steb.utils import RESULTS_DIR
from steb.validation import validate_all_configs


def add_common_arguments(parser):
    """Adds common arguments to the parser."""
    parser.add_argument("model_name_or_path", nargs="?", help="Model name (HF ID), or local path to the model.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for embedding.")
    parser.add_argument("--output-folder", default=RESULTS_DIR, help="Folder to save the results to.")
    parser.add_argument("--force-reload", default=False, action="store_true", help="Whether to force reload the datasets.")
    parser.add_argument("--progress-bar", default=False, action="store_true", help="Show a progress bar.")
    parser.add_argument("--seed", type=int, default=42, help="The random seed to use.")


def add_iteration_arguments(parser):
    """Adds iteration arguments (episode sizes, n per class) to the parser."""
    parser.add_argument("-e", "--episode-sizes", type=int, nargs="+", help="Number of atomic units to form writing sample.")
    parser.add_argument("--n-episodes-per-class", type=int, default=50, help="Number of examples per class.")


def run_validate():
    """Validates all dataset config.json files."""
    print(colored("Validating all dataset configs...", "cyan"))
    num_valid, num_invalid = validate_all_configs()
    print()
    print(f"Results: {num_valid} valid, {num_invalid} invalid")
    if num_invalid > 0:
        sys.exit(1)


def run_new_dataset(args):
    """Scaffolds a new dataset directory with a template config.json."""
    package_dir = os.path.dirname(os.path.abspath(__file__))
    steb_dir = os.path.dirname(package_dir)
    datasets_dir = os.path.join(steb_dir, "steb_datasets")
    dataset_dir = os.path.join(datasets_dir, args.name)

    if os.path.exists(dataset_dir):
        print(colored(f"Dataset directory already exists: {dataset_dir}", "red"))
        sys.exit(1)

    os.makedirs(dataset_dir)

    if args.type == "huggingface":
        config = {
            "dataset_name": args.name,
            "type": "huggingface",
            "record_handler": {
                "text_getter": "text",
                "label_getter": "label",
            },
            "loader_kwargs": {
                "path": args.name,
                "split": "train",
            },
            "tasks": {},
        }
    else:
        config = {
            "dataset_name": args.name,
            "type": "custom",
            "data_dir": args.name,
            "loader_function": f"load_{args.name.replace('-', '_')}_dataset",
            "record_handler": {
                "text_getter": "text",
                "label_getter": "label",
            },
            "tasks": {},
        }
        # Write a stub loader.py
        loader_path = os.path.join(dataset_dir, "loader.py")
        func_name = f"load_{args.name.replace('-', '_')}_dataset"
        with open(loader_path, "w") as f:
            f.write(f'from typing import Any, Dict, List\n\n\n')
            f.write(f'def {func_name}(data_dir: str) -> List[Dict[str, Any]]:\n')
            f.write(f'    """\n')
            f.write(f'    Loads the {args.name} dataset.\n\n')
            f.write(f'    Args:\n')
            f.write(f'        data_dir: Path to the raw data directory.\n\n')
            f.write(f'    Returns:\n')
            f.write(f'        A list of records with \'text\' and \'label\' fields.\n')
            f.write(f'    """\n')
            f.write(f'    raise NotImplementedError("TODO: implement loader")\n')

    config_path = os.path.join(dataset_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    print(colored(f"Created dataset scaffold at: {dataset_dir}", "green"))
    print(f"  - config.json")
    if args.type == "custom":
        print(f"  - loader.py (stub)")
    print()
    print("Next steps:")
    print("  1. Fill in the 'tasks' field in config.json")
    if args.type == "huggingface":
        print("  2. Update 'path' and 'split' in loader_kwargs")
        print("  3. Update 'text_getter' and 'label_getter' in record_handler")
    else:
        print("  2. Implement the loader function in loader.py")
        print("  3. Add the raw data to raw_datasets/" + args.name)
    print("  4. Run 'steb validate' to check your config")


def parse_new_dataset_args() -> argparse.Namespace:
    """
    Creates and parses arguments for the 'new-dataset' subcommand.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Scaffold a new STEB dataset.")
    parser.add_argument("cmd", help=argparse.SUPPRESS)
    parser.add_argument("name", help="Name for the new dataset.")
    parser.add_argument(
        "--type",
        choices=["huggingface", "custom"],
        default="huggingface",
        help="Dataset type.",
    )
    return parser.parse_args()


def create_preset_parser() -> argparse.ArgumentParser:
    """
    Creates the argument parser for the '--preset' mode.

    Returns:
        An ArgumentParser configured for preset evaluation.
    """
    parser = argparse.ArgumentParser(description="Run STEB with a preset configuration.")
    parser.add_argument(
        "--preset",
        type=str,
        required=True,
        help=f"Preset name. Available: {list(PRESETS.keys())}",
    )
    add_common_arguments(parser)
    return parser


def main():
    """
    The main function for the STEB CLI.
    Parses command-line arguments and runs the evaluation.
    """
    # Handle utility commands before evaluation parsing
    if len(sys.argv) >= 2 and sys.argv[1] == "validate":
        run_validate()
        return

    if len(sys.argv) >= 2 and sys.argv[1] == "new-dataset":
        args = parse_new_dataset_args()
        run_new_dataset(args)
        return

    if "--preset" in sys.argv:
        parser = create_preset_parser()
        args = parser.parse_args()

        if not args.model_name_or_path:
            parser.error("the following arguments are required: model_name_or_path")

        model = get_model(args.model_name_or_path)

        try:
            preset_config = resolve_preset(args.preset)
        except ValueError as e:
            parser.error(str(e))

        task_configs = preset_config["config"]["tasks"]
        print(f"Running preset: {args.preset}")
        print("Found #{:02d} evaluations in preset".format(len(task_configs)))
        for item in task_configs:
            task_name = item["task"]
            datasets = item["datasets"]
            current_episode_sizes = item["episode_sizes"]
            current_n_episodes = item["n_episodes_per_class"]
            
            evaluate(
                model,
                datasets,
                episode_sizes=current_episode_sizes,
                task_name=task_name,
                n_episodes_per_class=current_n_episodes,
                batch_size=args.batch_size,
                output_folder=args.output_folder,
                force_reload=args.force_reload,
                progress_bar=args.progress_bar,
                seed=args.seed,
                run_name=args.preset,
            )
        return

    # Standard mode parser
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="task", required=True)

    # Base parser for common arguments
    base_parser = argparse.ArgumentParser(add_help=False)
    add_common_arguments(base_parser)
    add_iteration_arguments(base_parser)

    # 'all' task parser
    all_parser = subparsers.add_parser("all", help="Run all tasks.", parents=[base_parser])
    all_parser.add_argument("--dataset", default=None, help="Dataset to evaluate on. If not specified, runs on all supported datasets.")

    # 'clustering' task parser
    clustering_parser = subparsers.add_parser("clustering", help="Run clustering task.", parents=[base_parser])
    clustering_parser.add_argument("--dataset", choices=get_supported_datasets("clustering"), default=None, help="Dataset to evaluate on. If not specified, runs on all supported datasets.")
    clustering_parser.add_argument("--list-datasets", action="store_true", help="List all available datasets for this task.")

    # 'all_to_all_pair_classification' task parser
    all_to_all_pair_classification_parser = subparsers.add_parser("all_to_all_pair_classification", help="Run all-to-all pair classification task.", parents=[base_parser])
    all_to_all_pair_classification_parser.add_argument("--dataset", choices=get_supported_datasets("all_to_all_pair_classification"), default=None, help="Dataset to evaluate on. If not specified, runs on all supported datasets.")
    all_to_all_pair_classification_parser.add_argument("--list-datasets", action="store_true", help="List all available datasets for this task.")

    # 'pre_defined_pair_classification' task parser
    pre_defined_pair_classification_parser = subparsers.add_parser("pre_defined_pair_classification", help="Run pre-defined pair classification task.", parents=[base_parser])
    pre_defined_pair_classification_parser.add_argument("--dataset", choices=get_supported_datasets("pre_defined_pair_classification"), default=None, help="Dataset to evaluate on. If not specified, runs on all supported datasets.")
    pre_defined_pair_classification_parser.add_argument("--list-datasets", action="store_true", help="List all available datasets for this task.")

    # 'order_alignment' task parser
    order_alignment_parser = subparsers.add_parser("order_alignment", help="Run order alignment task.", parents=[base_parser])
    order_alignment_parser.add_argument("--dataset", choices=get_supported_datasets("order_alignment"), default=None, help="Dataset to evaluate on. If not specified, runs on all supported datasets.")
    order_alignment_parser.add_argument("--list-datasets", action="store_true", help="List all available datasets for this task.")

    # 'retrieval' task parser
    retrieval_parser = subparsers.add_parser("retrieval", help="Run retrieval task.", parents=[base_parser])
    retrieval_parser.add_argument("--dataset", choices=get_supported_datasets("retrieval"), default=None, help="Dataset to evaluate on. If not specified, runs on all supported datasets.")
    retrieval_parser.add_argument("--list-datasets", action="store_true", help="List all available datasets for this task.")

    # 'probing' task parser
    probing_parser = subparsers.add_parser("probing", help="Run probing task.", parents=[base_parser])
    probing_parser.add_argument("--dataset", choices=get_supported_datasets("probing"), default=None, help="Dataset to evaluate on. If not specified, runs on all supported datasets.")
    probing_parser.add_argument("--list-datasets", action="store_true", help="List all available datasets for this task.")

    # Handle --list-datasets before full parsing to avoid required arg errors
    if "--list-datasets" in sys.argv:
        task = sys.argv[1]
        datasets = get_supported_datasets(task)
        print(f"Available datasets for {task}:")
        for dataset in datasets:
            print(f"- {dataset}")
        return

    args = parser.parse_args()

    if not args.model_name_or_path:
        parser.error("the following arguments are required: model_name_or_path")

    if args.task == "pre_defined_pair_classification":
        args.episode_sizes = [1]
        args.n_episodes_per_class = 2
    elif args.task == "probing":
        args.episode_sizes = [1]
        args.n_episodes_per_class = 1

    if not args.episode_sizes:
        parser.error("the following arguments are required: -e/--episode-sizes")

    model = get_model(args.model_name_or_path)

    if args.task == "all":
        datasets = get_all_datasets() if not args.dataset else [args.dataset]
        task_name = None
    else:
        datasets = get_supported_datasets(args.task) if not args.dataset else [args.dataset]
        task_name = args.task

    evaluate(
        model,
        datasets,
        episode_sizes=args.episode_sizes,
        task_name=task_name,
        n_episodes_per_class=args.n_episodes_per_class,
        batch_size=args.batch_size,
        output_folder=args.output_folder,
        force_reload=args.force_reload,
        progress_bar=args.progress_bar,
        seed=args.seed,
        run_name=args.task,
    )

if __name__ == "__main__":
    main()
