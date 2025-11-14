
import argparse
from steb import get_model, get_datasets, evaluate

def main():
    """
    The main function for the STEB CLI.
    Parses command-line arguments and runs the evaluation.
    """
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run the STEB evaluation.")
    run_parser.add_argument(
        "-m", "--model-name-or-path",
        dest="model_name_or_path",
        default="rrivera1849/LUAR-CRUD",
        help="Model name (HF ID), or local path to the model."
    )
    run_parser.add_argument(
        "-t", "--tasks",
        dest="datasets",
        nargs="+",
        default=None,
        help="List of datasets to evaluate on. If not specified, runs on all available datasets."
    )
    run_parser.add_argument(
        "-e", "--episode-sizes",
        dest="episode_sizes",
        type=int,
        nargs="+",
        required=True,
        help="Number of atomic units to form writing sample."
    )
    run_parser.add_argument(
        "--n-episodes-per-class",
        dest="n_episodes_per_class",
        type=int,
        default=50,
        help="Number of examples per class."
    )
    run_parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=32,
        help="Batch size for embedding."
    )
    run_parser.add_argument(
        "--output-folder",
        dest="output_folder",
        default="results",
        help="Folder to save the results to."
    )
    run_parser.add_argument(
        "--force-reload",
        dest="force_reload",
        default=False,
        action="store_true",
        help="Whether to force reload the datasets."
    )
    run_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The random seed to use."
    )
    args = parser.parse_args()

    if args.command == "run":
        model = get_model(args.model_name_or_path)
        datasets = get_datasets(args.datasets)

        evaluate(
            model,
            datasets,
            episode_sizes=args.episode_sizes,
            n_episodes_per_class=args.n_episodes_per_class,
            batch_size=args.batch_size,
            output_folder=args.output_folder,
            force_reload=args.force_reload,
            seed=args.seed,
        )

if __name__ == "__main__":
    main()
