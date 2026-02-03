import argparse
import sys
from steb import get_model, get_all_datasets, get_supported_datasets, evaluate
from steb.utils import RESULTS_DIR

from steb.presets import resolve_preset, PRESETS

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


def main():
    """
    The main function for the STEB CLI.
    Parses command-line arguments and runs the evaluation.
    """
    if "--preset" in sys.argv:
        # Preset mode parser
        parser = argparse.ArgumentParser(description="Run STEB with a preset configuration.")
        parser.add_argument("--preset", type=str, required=True, help=f"Preset name. Available: {list(PRESETS.keys())}")
        add_common_arguments(parser)
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

    retrieval_parser = subparsers.add_parser("retrieval", help="Run retrieval task.", parents=[base_parser])
    retrieval_parser.add_argument("--dataset", choices=get_supported_datasets("retrieval"), default=None, help="Dataset to evaluate on. If not specified, runs on all supported datasets.")
    retrieval_parser.add_argument("--list-datasets", action="store_true", help="List all available datasets for this task.")

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
    )

if __name__ == "__main__":
    main()
