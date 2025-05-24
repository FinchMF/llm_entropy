"""Configuration handling for llm_entropy package."""

import os
import yaml
from typing import Dict, Any

def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from yaml file.
    
    Args:
        config_path: Path to custom config file. If None, loads default config.
    
    Returns:
        Dictionary containing configuration parameters.
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'default.yaml')
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config
