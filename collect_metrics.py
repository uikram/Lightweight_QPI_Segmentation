import shutil
from pathlib import Path


def collect_and_rename_metrics():
    results_dir = Path("results")
    output_dir = results_dir / "rank_wise_result"

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Created/Found output directory: {output_dir}\n")

    count = 0

    # Find all metrics.json files
    for metrics_file in results_dir.rglob("metrics.json"):

        # Skip files inside output directory
        if output_dir in metrics_file.parents:
            continue

        try:
            # Get folder name directly under results/
            relative_path = metrics_file.relative_to(results_dir)

            # Example:
            # edge_sam_lora_r4/metrics.json
            # -> edge_sam_lora_r4
            model_and_rank = relative_path.parts[0]

            # Create new filename
            new_filename = f"{model_and_rank}_metrics.json"

            destination = output_dir / new_filename

            # Copy file
            shutil.copy2(metrics_file, destination)

            print(f"Copied:")
            print(f"  {metrics_file}")
            print(f"  --> {destination}\n")

            count += 1

        except Exception as e:
            print(f"Skipping {metrics_file}: {e}")

    print(f"Done! Successfully processed {count} metric files.")


if __name__ == "__main__":
    collect_and_rename_metrics()