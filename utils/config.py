"""
Minimal configuration loader with validation.
Replaces complex class-based configs with SimpleNamespace.
Flattens nested YAML sections for compatibility with model classes.
"""
import yaml
import torch
from types import SimpleNamespace
from pathlib import Path

def load_config_from_yaml(yaml_path: str, model_name: str = None) -> SimpleNamespace:
    """
    Load config from YAML, flattening sections like 'model', 'data', etc.
    to ensure attributes are accessible at the top level.
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    flat_data = {}
    
    # Pre-defined sections that should be flattened
    # This matches the structure of your YAML files
    sections_to_flatten = [
        'model', 
        'system', 
        'data', 
        'training', 
        'evaluation', 
        'dataset', 
        'optimizer',
        'eval_modes' 
    ]
    
    # 1. Flatten the dictionary
    for key, value in data.items():
        if key in sections_to_flatten and isinstance(value, dict):
            # Lift keys from the section to the top level
            for sub_key, sub_value in value.items():
                flat_data[sub_key] = sub_value
        else:
            # Keep top-level keys as is
            flat_data[key] = value

    # 2. Recursive conversion to SimpleNamespace
    def to_namespace(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: to_namespace(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [to_namespace(item) for item in d]
        return d

    config = to_namespace(flat_data)

    # 3. Apply Defaults & Path Conversions
    
    # Hardware
    if not hasattr(config, 'device'):
        config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not hasattr(config, 'num_workers'):
        config.num_workers = 4
        
    # Training defaults
    if not hasattr(config, 'batch_size'):
        config.batch_size = 16
    if not hasattr(config, 'num_epochs'):
        config.num_epochs = 10
        
    # Alias handling: Ensure consistent naming across models
    # CLIP uses 'model_name', LoRA uses 'model_id'
    if hasattr(config, 'model_id') and not hasattr(config, 'model_name'):
        config.model_name = config.model_id
    if hasattr(config, 'model_name') and not hasattr(config, 'model_id'):
        config.model_id = config.model_name
        
    # Paths (Auto-convert specific keys to Path objects)
    path_keys = [
        'results_dir', 'checkpoint_dir', 'cache_dir', 'output_dir', 
        'plots_dir', 'data_root', 'image_dir', 'annotation_file',
        'train_image_dir', 'train_file', 'val_image_dir', 'val_file'
    ]
    
    for key in path_keys:
        if hasattr(config, key):
            val = getattr(config, key)
            if val is not None:
                setattr(config, key, Path(val))
            
    # Evaluation
    if not hasattr(config, 'k_shots') or config.k_shots is None:
        config.k_shots = [1, 2, 4, 8, 16]

    return config