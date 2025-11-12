
import json
import os
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt

def find_metrics_files():
    """
    Finds all metrics.json files in the outputs directory and parses their paths.

    Returns:
        A list of dictionaries, where each dictionary contains the path to a
        metrics.json file and the parsed information from its path.
    """
    outputs_dir = Path("outputs")
    metrics_files = []
    for path in outputs_dir.rglob("metrics.json"):
        parts = path.parts
        if len(parts) >= 5:
            try:
                dataset = parts[1]
                model = parts[2]
                episode_info = parts[3].split("_")
                episode_size = int(episode_info[0])
                n_episodes_per_class = int(episode_info[1])
                task = parts[4]
                metrics_files.append({
                    "path": path,
                    "dataset": dataset,
                    "model": model,
                    "episode_size": episode_size,
                    "n_episodes_per_class": n_episodes_per_class,
                    "task": task,
                })
            except (IndexError, ValueError) as e:
                print(f"Skipping malformed path: {path} ({e})")
    return metrics_files

def process_metrics(metrics_files):
    """
    Processes the collected metrics, grouping them by task, episode size, and dataset.

    Args:
        metrics_files: A list of dictionaries, as returned by find_metrics_files.

    Returns:
        A dictionary containing the processed and aggregated metrics.
    """
    processed_data = defaultdict(lambda: {
        "aggregate": defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
        "individual": defaultdict(list)
    })

    for file_info in metrics_files:
        with open(file_info["path"]) as f:
            metrics = json.load(f)

        task = file_info["task"]
        model = file_info["model"]
        dataset = file_info["dataset"]
        episode_size = file_info["episode_size"]
        n_episodes = file_info["n_episodes_per_class"]
        agg_key = f"e{episode_size}_n{n_episodes}"

        # Store individual metrics, grouped by dataset
        processed_data[task]["individual"][dataset].append({
            "model": model,
            "episode_size": episode_size,
            "n_episodes_per_class": n_episodes,
            "metrics": metrics,
        })

        # Collect metrics for aggregation
        for metric_name, value in metrics.items():
            processed_data[task]["aggregate"][agg_key][model][metric_name].append(value)

    # Calculate aggregate metrics
    for task in processed_data:
        for agg_key in processed_data[task]["aggregate"]:
            for model in processed_data[task]["aggregate"][agg_key]:
                for metric_name, values in processed_data[task]["aggregate"][agg_key][model].items():
                    processed_data[task]["aggregate"][agg_key][model][metric_name] = sum(values) / len(values)

    return processed_data

def create_aggregate_table(task_name, aggregate_data):
    """
    Creates pretty-printed tables for the aggregate metrics, one for each agg_key.

    Args:
        task_name: The name of the task.
        aggregate_data: A dictionary of aggregated metrics for the task.

    Returns:
        A list of strings, each containing a formatted table.
    """
    tables = []
    sorted_agg_keys = sorted(aggregate_data.keys(), key=lambda x: int(x.split('_')[0][1:]))

    for agg_key in sorted_agg_keys:
        data = aggregate_data[agg_key]
        if not data:
            continue

        primary_metric = "v_measure" if task_name == "clustering" else "eer"
        sorted_models = sorted(
            data.items(),
            key=lambda item: item[1].get(primary_metric, 0),
            reverse=task_name == "clustering",  # v_measure is higher is better
        )

        header = f"| {'Model':<30} |"
        all_metric_names = sorted(list(set(m for model_metrics in data.values() for m in model_metrics.keys())))
        for name in all_metric_names:
            header += f" {name:<15} |"

        separator = "-" * len(header)
        table = [f"### Aggregate Metrics for {task_name.title()} ({agg_key})", header, separator]

        for model, metrics in sorted_models:
            row = f"| {model:<30} |"
            for name in all_metric_names:
                value = metrics.get(name, "N/A")
                if isinstance(value, float):
                    row += f" {value:<15.4f} |"
                else:
                    row += f" {str(value):<15} |"
            table.append(row)

        tables.append("\n".join(table))

    return tables

def create_individual_table(task_name, individual_data):
    """
    Creates pretty-printed tables for the individual metrics, one for each dataset and configuration.

    Args:
        task_name: The name of the task.
        individual_data: A dictionary of individual metrics for the task, grouped by dataset.

    Returns:
        A list of strings, each containing a formatted table.
    """
    tables = []
    for dataset, data in individual_data.items():
        if not data:
            continue

        # Group data by episode size and n_episodes
        configs = defaultdict(list)
        for item in data:
            configs[(item["episode_size"], item["n_episodes_per_class"])].append(item)

        for (episode_size, n_episodes), config_data in sorted(configs.items()):
            primary_metric = "v_measure" if task_name == "clustering" else "eer"
            sorted_data = sorted(
                config_data,
                key=lambda item: item["metrics"].get(primary_metric, 0),
                reverse=task_name == "clustering",
            )

            all_metric_names = sorted(list(set(m for item in config_data for m in item["metrics"].keys())))
            header = f"| {'Model':<30} |"
            for name in all_metric_names:
                header += f" {name:<15} |"

            separator = "-" * len(header)
            table = [f"### Individual Metrics for {task_name.title()} on {dataset} (e{episode_size}_n{n_episodes})", header, separator]

            for item in sorted_data:
                row = f"| {item['model']:<30} |"
                for name in all_metric_names:
                    value = item["metrics"].get(name, "N/A")
                    if isinstance(value, float):
                        row += f" {value:<15.4f} |"
                    else:
                        row += f" {str(value):<15} |"
                table.append(row)

            tables.append("\n".join(table))

    return tables

def create_plots(task_name, processed_data, output_dir):
    """
    Creates plots for the processed metrics.

    Args:
        task_name: The name of the task.
        processed_data: A dictionary of processed metrics for the task.
        output_dir: The directory to save the plots in.
    """
    # Aggregate plots
    agg_data = processed_data["aggregate"]
    models = sorted(list(set(model for agg_key in agg_data for model in agg_data[agg_key])))
    all_metric_names = sorted(list(set(m for agg_key in agg_data for model in agg_data[agg_key] for m in agg_data[agg_key][model])))

    for metric_name in all_metric_names:
        plt.figure()
        for model in models:
            episode_sizes = []
            metric_values = []
            for agg_key in sorted(agg_data.keys(), key=lambda x: int(x.split('_')[0][1:])):
                if model in agg_data[agg_key] and metric_name in agg_data[agg_key][model]:
                    episode_size = int(agg_key.split('_')[0][1:])
                    episode_sizes.append(episode_size)
                    metric_values.append(agg_data[agg_key][model][metric_name])
            plt.plot(episode_sizes, metric_values, marker='o', linestyle='-', label=model)

        plt.title(f"Task: {task_name.title()}")
        plt.xlabel("Episode Size")
        plt.ylabel(metric_name.replace("_", " ").title())
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir / f"aggregate_{task_name}_{metric_name}.png")
        plt.close()

    # Individual plots
    ind_data = processed_data["individual"]
    for dataset, dataset_data in ind_data.items():
        models = sorted(list(set(item['model'] for item in dataset_data)))
        all_metric_names = sorted(list(set(m for item in dataset_data for m in item['metrics'])))

        for metric_name in all_metric_names:
            plt.figure()
            for model in models:
                episode_sizes = []
                metric_values = []
                sorted_data = sorted(dataset_data, key=lambda x: x['episode_size'])
                for item in sorted_data:
                    if item['model'] == model and metric_name in item['metrics']:
                        episode_sizes.append(item['episode_size'])
                        metric_values.append(item['metrics'][metric_name])
                plt.plot(episode_sizes, metric_values, marker='o', linestyle='-', label=model)

            plt.title(f"Task: {task_name.title()} on {dataset}")
            plt.xlabel("Episode Size")
            plt.ylabel(metric_name.replace("_", " ").title())
            plt.legend()
            plt.grid(True)
            plt.savefig(output_dir / f"individual_{task_name}_{dataset}_{metric_name}.png")
            plt.close()

def main():
    """
    Main function to find, process, and print metrics.
    """
    output_dir = Path("outputs") / "pretty-print"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_files = find_metrics_files()
    processed_data = process_metrics(metrics_files)

    text_output = []
    json_output = {}

    for task, data in processed_data.items():
        aggregate_tables = create_aggregate_table(task, data["aggregate"])
        text_output.extend(aggregate_tables)
        for table in aggregate_tables:
            print(table)
            print("\n")

        individual_tables = create_individual_table(task, data["individual"])
        text_output.extend(individual_tables)
        for table in individual_tables:
            print(table)
            print("\n")

        create_plots(task, data, output_dir)
        json_output[task] = data

    with open(output_dir / "metrics_summary.txt", "w") as f:
        f.write("\n\n".join(text_output))

    with open(output_dir / "metrics_summary.json", "w") as f:
        json.dump(json_output, f, indent=2)

    print(f"Metrics summary saved to {output_dir}")

if __name__ == "__main__":
    main()
