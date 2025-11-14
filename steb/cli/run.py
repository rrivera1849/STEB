
import argparse
import sys
from steb import get_model, get_all_datasets, get_supported_datasets, evaluate

def main():
    """
    The main function for the STEB CLI.
    Parses command-line arguments and runs the evaluation.
    """
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="task", required=True)

    # Base parser for common arguments
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("model_name_or_path", nargs="?", help="Model name (HF ID), or local path to the model.")
    base_parser.add_argument("-e", "--episode-sizes", type=int, nargs="+", help="Number of atomic units to form writing sample.")
    base_parser.add_argument("--n-episodes-per-class", type=int, default=50, help="Number of examples per class.")
    base_parser.add_argument("--batch-size", type=int, default=32, help="Batch size for embedding.")
    base_parser.add_argument("--output-folder", default="results", help="Folder to save the results to.")
    base_parser.add_argument("--force-reload", default=False, action="store_true", help="Whether to force reload the datasets.")
    base_parser.add_argument("--progress-bar", default=False, action="store_true", help="Show a progress bar.")
    base_parser.add_argument("--seed", type=int, default=42, help="The random seed to use.")

    # 'all' task parser
    all_parser = subparsers.add_parser("all", help="Run all tasks.", parents=[base_parser])
    all_parser.add_argument("--dataset", default=None, help="Dataset to evaluate on. If not specified, runs on all supported datasets.")

    # 'clustering' task parser
    clustering_parser = subparsers.add_parser("clustering", help="Run clustering task.", parents=[base_parser])
    clustering_parser.add_argument("--dataset", choices=get_supported_datasets("clustering"), default=None, help="Dataset to evaluate on. If not specified, runs on all supported datasets.")
    clustering_parser.add_argument("--list-datasets", action="store_true", help="List all available datasets for this task.")

    # 'pair_classification' task parser
    pair_classification_parser = subparsers.add_parser("pair_classification", help="Run pair classification task.", parents=[base_parser])
    pair_classification_parser.add_argument("--dataset", choices=get_supported_datasets("pair_classification"), default=None, help="Dataset to evaluate on. If not specified, runs on all supported datasets.")
    pair_classification_parser.add_argument("--list-datasets", action="store_true", help="List all available datasets for this task.")

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
