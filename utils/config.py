"""
Configuration loader with validation.
Flattens nested YAML sections for compatibility with model/trainer classes.
"""
import yaml
import torch
from types import SimpleNamespace
from pathlib import Path


def load_config_from_yaml(yaml_path: str, model_name: str = None) -> SimpleNamespace:
    """
    Load config from YAML, flattening specified sections to top-level attributes.
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    flat_data = {}

    # All sections whose keys are lifted to the top level.
    # 'lora' is included so lora_r, insertion_strategy etc. are accessible
    # directly as config.lora_r, config.insertion_strategy.
    sections_to_flatten = [
        'model',
        'system',
        'data',
        'training',
        'evaluation',
        'dataset',
        'optimizer',
        'eval_modes',
        'lora',          # ← required for LoRA injection in get_model()
        'loss_weights',  # ← lifts lambda1_pmc, lambda2_bga, lambda3_pv
    ]

    for key, value in data.items():
        if key in sections_to_flatten and isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat_data[sub_key] = sub_value
        else:
            flat_data[key] = value

    def to_namespace(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: to_namespace(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [to_namespace(item) for item in d]
        return d

    config = to_namespace(flat_data)

    # ── Hardware defaults ──────────────────────────────────────────────────────
    if not hasattr(config, 'device'):
        config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not hasattr(config, 'num_workers'):
        config.num_workers = 4

    # ── Training defaults ──────────────────────────────────────────────────────
    if not hasattr(config, 'batch_size'):
        config.batch_size = 8
        
    if hasattr(config, 'epochs') and not hasattr(config, 'num_epochs'):
        config.num_epochs = config.epochs
    elif hasattr(config, 'num_epochs') and not hasattr(config, 'epochs'):
        config.epochs = config.num_epochs
    elif not hasattr(config, 'num_epochs') and not hasattr(config, 'epochs'):
        config.num_epochs = 50
        config.epochs = 50

    if not hasattr(config, 'num_classes'):
        config.num_classes = 5

    # ── Model name aliases ─────────────────────────────────────────────────────
    if hasattr(config, 'model_id') and not hasattr(config, 'model_name'):
        config.model_name = config.model_id
    if hasattr(config, 'model_name') and not hasattr(config, 'model_id'):
        config.model_id = config.model_name

    # ── Path conversion ────────────────────────────────────────────────────────
    path_keys = [
        'results_dir', 'checkpoint_dir', 'cache_dir', 'output_dir',
        'plots_dir', 'data_root', 'image_dir', 'annotation_file',
        'train_image_dir', 'train_file', 'val_image_dir', 'val_file',
    ]
    for key in path_keys:
        if hasattr(config, key):
            val = getattr(config, key)
            if val is not None:
                setattr(config, key, Path(val))

    # ── Safe fallbacks for paths used in trainer ───────────────────────────────
    # These prevent AttributeError when results_dir / checkpoint_dir are absent
    # from the YAML. The trainer will create the directory automatically.
    if not hasattr(config, 'results_dir'):
        config.results_dir = Path('results')
    if not hasattr(config, 'checkpoint_dir'):
        config.checkpoint_dir = config.results_dir / 'checkpoints'

    # ── Misc defaults ──────────────────────────────────────────────────────────
    if not hasattr(config, 'k_shots') or config.k_shots is None:
        config.k_shots = [1, 2, 4, 8, 16]

    return config
