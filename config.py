"""Configuration handling for llm_entropy package."""

import os
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class AnalysisConfig:
    temperature: float = 1.0
    use_sampling: bool = False
    num_sentences: int = 10

@dataclass
class Config:
    # System settings
    batch_size: int = 8
    max_workers: int = 4
    model_timeout: int = 300
    max_retries: int = 3
    cache_dir: str = ".cache"
    output_dir: str = "outputs"
    
    # Analysis settings
    analysis: AnalysisConfig = AnalysisConfig()
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'Config':
        if not os.path.exists(config_path):
            return cls()
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Handle nested analysis config
        if 'analysis' in config_dict:
            analysis_dict = config_dict.pop('analysis')
            config_dict['analysis'] = AnalysisConfig(**analysis_dict)
            
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'batch_size': self.batch_size,
            'max_workers': self.max_workers,
            'model_timeout': self.model_timeout,
            'max_retries': self.max_retries,
            'cache_dir': self.cache_dir,
            'output_dir': self.output_dir,
            'analysis': {
                'temperature': self.analysis.temperature,
                'use_sampling': self.analysis.use_sampling,
                'num_sentences': self.analysis.num_sentences
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        if '.' in key:
            section, param = key.split('.', 1)
            section_obj = getattr(self, section, None)
            if section_obj is not None:
                return getattr(section_obj, param, default)
        return getattr(self, key, default)

def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from a YAML file or return default config.
    
    Args:
        config_path: Path to config YAML file. If None, looks for config.yml in current directory
        
    Returns:
        Config object with either loaded or default settings
    """
    default_paths = [
        os.path.join(os.getcwd(), "config.yml"),
        os.path.join(os.path.dirname(__file__), "config.yml")
    ]
    
    if config_path is None:
        # Try default locations
        for path in default_paths:
            if os.path.exists(path):
                config_path = path
                break
    
    if config_path and os.path.exists(config_path):
        try:
            return Config.from_yaml(config_path)
        except Exception as e:
            raise ValueError(f"Failed to load config from {config_path}: {str(e)}")
    
    # Return default config if no valid config file found
    return Config()
