import json
import pandas as pd
from pathlib import Path
import argparse
import re

def compile_all_results(results_dir="results"):
    results_path = Path(results_dir)
    
    metrics_dict = {"ablation": [], "full_finetune": [], "lora_sweep": []}
    morph_dict = {"ablation": [], "full_finetune": [], "lora_sweep": []}
    
    # Known loss configurations from the directory structure
    known_losses = ["dice_only", "full", "pmc_bga", "pmc"]
    
    def get_category(path_obj):
        path_str = str(path_obj).lower()
        if "ablation" in path_str: 
            return "ablation"
        if "full_finetune" in path_str or "fullft" in path_str: 
            return "full_finetune"
        return "lora_sweep"

    def get_loss_config(path_obj):
        # Scan the path parts for the loss folder name
        for part in path_obj.parts:
            if part.lower() in known_losses:
                return part.lower()
        return "N/A"

    print(f"Scanning {results_path} for metrics.json files...")
    for metrics_file in results_path.rglob("metrics.json"):
        try:
            category = get_category(metrics_file)
            loss_config = get_loss_config(metrics_file)
            
            with open(metrics_file, 'r') as f:
                data = json.load(f)

            model_name = data.get("model_name", "UNKNOWN")
            
            if "_R" in model_name:
                parts = model_name.rsplit("_R", 1)
                arch, rank = parts[0], parts[1]
            elif "_FULL" in model_name:
                arch, rank = model_name.replace("_FULL", ""), "0"
            else:
                arch, rank = model_name, "0"
                
            if category == "ablation" and rank == "0":
                rank = "8" 

            params = data.get("parameters", {})
            seg_metrics = data.get("seg_metrics", {}).get("val_best", {})
            
            raw_row = {
                "Architecture": arch,
                "Category": category.upper(),
                "Loss Config": loss_config,
                "Rank": rank,
                "Trainable %": params.get("trainable_percentage"),
                "Trainable Params": params.get("trainable_parameters"),
                "Total Params": params.get("total_parameters")
            }
            raw_row.update(seg_metrics)

            # Sanitize: Convert any nested lists/dicts to strings to prevent Pandas C-API crashes
            clean_row = {}
            for k, v in raw_row.items():
                if isinstance(v, (list, dict)):
                    clean_row[k] = str(v)
                else:
                    clean_row[k] = v

            metrics_dict[category].append(clean_row)
            
        except Exception as e:
            print(f"[Warning] Could not parse {metrics_file}: {e}")

    print(f"\nScanning {results_path} for morphology_trends*.csv files...")
    for csv_file in results_path.rglob("morphology_trends*.csv"):
        try:
            category = get_category(csv_file)
            loss_config = get_loss_config(csv_file)
            
            match = re.search(r"rank_(\d+)", csv_file.name)
            rank = match.group(1) if match else "0"
            
            if category == "ablation" and rank == "0":
                rank = "8"

            if csv_file.parent.name == "default_run":
                model_dir_name = csv_file.parent.parent.name
            else:
                model_dir_name = csv_file.parent.name

            if "_R" in model_dir_name:
                arch = model_dir_name.rsplit("_R", 1)[0]
            elif "_r" in model_dir_name:
                arch = model_dir_name.rsplit("_r", 1)[0]
            elif "_FULL" in model_dir_name:
                arch = model_dir_name.replace("_FULL", "")
            else:
                arch = model_dir_name

            df = pd.read_csv(csv_file)
            df.insert(0, "Rank", rank)
            df.insert(0, "Loss Config", loss_config)
            df.insert(0, "Category", category.upper())
            df.insert(0, "Architecture", arch)
            
            morph_dict[category].append(df)
            
        except Exception as e:
            print(f"[Warning] Could not parse {csv_file}: {e}")

    def save_metrics(data_list, name_base):
        if not data_list: return
        df = pd.DataFrame(data_list)
        df['Rank_Num'] = pd.to_numeric(df['Rank'], errors='coerce')
        df = df.sort_values(by=['Architecture', 'Loss Config', 'Rank_Num']).drop(columns=['Rank_Num'])
        
        # Added Loss Config to headline columns
        headline_cols = ["Architecture", "Category", "Loss Config", "Rank", "Trainable %", "mean_dice", "mean_iou", "aji", "bf1", "phase_vol_error"]
        ordered_cols = [c for c in headline_cols if c in df.columns] + [c for c in df.columns if c not in headline_cols]
        df = df[ordered_cols]
        
        df.to_csv(results_path / f"{name_base}.csv", index=False)
        df.to_json(results_path / f"{name_base}.json", orient="records", indent=4)
        print(f" -> Saved Metrics: {name_base} ({len(data_list)} models)")

    def save_morph(data_list, name_base):
        if not data_list: return
        compiled_df = pd.concat(data_list, ignore_index=True)
        compiled_df['Rank_Num'] = pd.to_numeric(compiled_df['Rank'], errors='coerce')
        
        sort_cols = ['Architecture', 'Loss Config', 'Rank_Num']
        if 'storage_day' in compiled_df.columns: sort_cols.append('storage_day')
        if 'stem' in compiled_df.columns: sort_cols.append('stem')
            
        compiled_df = compiled_df.sort_values(by=sort_cols).drop(columns=['Rank_Num'])
        compiled_df.to_csv(results_path / f"{name_base}.csv", index=False)
        compiled_df.to_json(results_path / f"{name_base}.json", orient="records", indent=4)
        print(f" -> Saved Morphology: {name_base} ({len(data_list)} files compiled)")

    print("\n--- Compiling Outputs ---")
    for cat in ["ablation", "full_finetune", "lora_sweep"]:
        save_metrics(metrics_dict[cat], f"{cat}_metrics")
        save_morph(morph_dict[cat], f"{cat}_morphology")
    print("\n[SUCCESS] All data categorized and compiled.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile all QPI results into distinct CSV/JSON files.")
    parser.add_argument("--results_dir", default="results", help="Root directory containing all results.")
    args = parser.parse_args()
    
    compile_all_results(args.results_dir)