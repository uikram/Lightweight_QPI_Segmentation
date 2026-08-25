import yaml
from pathlib import Path

ARCH_CONFIGS = {
    "mobilenet_unet": {
        "image_size": 256,
        "insertion_strategy": "encoder_only",
        "batch_size": 16,
        "learning_rate": 0.001,
    },
    "mobile_sam": {
        "image_size": 1024,
        "insertion_strategy": "attention_blocks",
        "batch_size": 8,
        "learning_rate": 0.0005,
    },
    "edge_sam": {
        "image_size": 256,
        "insertion_strategy": "encoder_only",
        "batch_size": 16,
        "learning_rate": 0.0005,
    },
}

ABLATIONS = {
    "dice_only":  {"loss_type": "dice_only"},
    "pmc":        {"loss_type": "physics_aware", "lambda1_pmc": 0.1, "lambda2_bga": 0.0,  "lambda3_pv": 0.0},
    "pmc_bga":    {"loss_type": "physics_aware", "lambda1_pmc": 0.1, "lambda2_bga": 0.05, "lambda3_pv": 0.0},
    "full":       {"loss_type": "physics_aware", "lambda1_pmc": 0.1, "lambda2_bga": 0.05, "lambda3_pv": 0.1},
}

CLASS_WEIGHTS = [0.5, 1.0, 1.5, 2.0, 2.0]

for arch, ap in ARCH_CONFIGS.items():
    for ab_name, ab in ABLATIONS.items():
        results_dir = f"results/ablation/{arch}/{ab_name}"
        config = {
            "experiment_name": f"ablation_{arch}_{ab_name}",
            "data_root": "./dataset",
            "cache_dir": "./cache",
            "dataset": {
                "batch_size": ap["batch_size"],
                "num_workers": 4,
                "image_size": ap["image_size"],
            },
            "model": {
                "architecture": arch,
                "in_channels": 1,
                "num_classes": 5,
            },
            "lora": {
                "lora_r": 8,
                "lora_alpha": 8.0,
                "lora_dropout": 0.0,
                "insertion_strategy": ap["insertion_strategy"],
            },
            "training": {
                "epochs": 50,
                "learning_rate": ap["learning_rate"],
                "optimizer": "AdamW",
                "mixed_precision": "fp16",
            },
            "loss_type": ab["loss_type"],
            "class_weights": CLASS_WEIGHTS,
            "results_dir": results_dir,
            "checkpoint_dir": f"{results_dir}/checkpoints",
        }
        if ab["loss_type"] == "physics_aware":
            config["lambda1_pmc"] = ab["lambda1_pmc"]
            config["lambda2_bga"] = ab["lambda2_bga"]
            config["lambda3_pv"]  = ab["lambda3_pv"]

        out_path = Path(f"configs/ablation/{arch}/{ab_name}.yaml")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            yaml.dump(config, f, sort_keys=False)
        print(f"  Written: {out_path}")

print("\n[Done] 12 ablation YAMLs generated.")