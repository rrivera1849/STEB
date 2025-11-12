
import json
import os
from collections import defaultdict
from pathlib import Path

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
    Processes the collected metrics, grouping them by task and calculating
    aggregate scores.

    Args:
        metrics_files: A list of dictionaries, as returned by find_metrics_files.

    Returns:
        A dictionary containing the processed and aggregated metrics.
    """
    processed_data = defaultdict(lambda: {
        "aggregate": defaultdict(lambda: defaultdict(list)),
        "individual": []
    })

    for file_info in metrics_files:
        with open(file_info["path"]) as f:
            metrics = json.load(f)

        task = file_info["task"]
        model = file_info["model"]
        dataset = file_info["dataset"]

        # Store individual metrics
        processed_data[task]["individual"].append({
            "model": model,
            "dataset": dataset,
            "metrics": metrics,
        })

        # Collect metrics for aggregation
        for metric_name, value in metrics.items():
            processed_data[task]["aggregate"][model][metric_name].append(value)

    # Calculate aggregate metrics
    for task in processed_data:
        for model in processed_data[task]["aggregate"]:
            for metric_name, values in processed_data[task]["aggregate"][model].items():
                processed_data[task]["aggregate"][model][metric_name] = sum(values) / len(values)

    return processed_data

def create_aggregate_table(task_name, aggregate_data):
    """
    Creates a pretty-printed table for the aggregate metrics.

    Args:
        task_name: The name of the task.
        aggregate_data: A dictionary of aggregated metrics for the task.

    Returns:
        A string containing the formatted table.
    """
    if not aggregate_data:
        return ""

    primary_metric = "v_measure" if task_name == "clustering" else "eer"
    sorted_models = sorted(
        aggregate_data.items(),
        key=lambda item: item[1].get(primary_metric, 0),
        reverse=task_name == "clustering",  # v_measure is higher is better
    )

    header = f"| {'Model':<30} |"
    all_metric_names = sorted(list(set(m for model_metrics in aggregate_data.values() for m in model_metrics.keys())))
    for name in all_metric_names:
        header += f" {name:<15} |"

    separator = "-" * len(header)
    table = [f"### Aggregate Metrics for {task_name.title()}", header, separator]

    for model, metrics in sorted_models:
        row = f"| {model:<30} |"
        for name in all_metric_names:
            value = metrics.get(name, "N/A")
            if isinstance(value, float):
                row += f" {value:<15.4f} |"
            else:
                row += f" {str(value):<15} |"
        table.append(row)

    return "\n".join(table)

def create_individual_table(task_name, individual_data):
    """
    Creates a pretty-printed table for the individual metrics.

    Args:
        task_name: The name of the task.
        individual_data: A list of individual metrics for the task.

    Returns:
        A string containing the formatted table.
    """
    if not individual_data:
        return ""

    primary_metric = "v_measure" if task_name == "clustering" else "eer"
    sorted_data = sorted(
        individual_data,
        key=lambda item: item["metrics"].get(primary_metric, 0),
        reverse=task_name == "clustering",
    )

    all_metric_names = sorted(list(set(m for item in individual_data for m in item["metrics"].keys())))
    header = f"| {'Model':<30} | {'Dataset':<30} |"
    for name in all_metric_names:
        header += f" {name:<15} |"

    separator = "-" * len(header)
    table = [f"### Individual Metrics for {task_name.title()}", header, separator]

    for item in sorted_data:
        row = f"| {item['model']:<30} | {item['dataset']:<30} |"
        for name in all_metric_names:
            value = item["metrics"].get(name, "N/A")
            if isinstance(value, float):
                row += f" {value:<15.4f} |"
            else:
                row += f" {str(value):<15} |"
        table.append(row)

    return "\n".join(table)

def main():
    """
    Main function to find, process, and print metrics.
    """
    metrics_files = find_metrics_files()
    processed_data = process_metrics(metrics_files)

    # Prepare outputs
    text_output = []
    json_output = {}

    for task, data in processed_data.items():
        aggregate_table = create_aggregate_table(task, data["aggregate"])
        individual_table = create_individual_table(task, data["individual"])

        # Console output
        print(aggregate_table)
        print("\n")
        print(individual_table)
        print("\n\n")

        # Text file output
        text_output.append(aggregate_table)
        text_output.append("\n")
        text_output.append(individual_table)
        text_output.append("\n\n")

        # JSON file output
        json_output[task] = data

    # Write to files
    with open("metrics_summary.txt", "w") as f:
        f.write("\n".join(text_output))

    with open("metrics_summary.json", "w") as f:
        json.dump(json_output, f, indent=2)

    print("Metrics summary saved to metrics_summary.txt and metrics_summary.json")

if __name__ == "__main__":
    main()
