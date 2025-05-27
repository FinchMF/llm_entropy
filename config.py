"""Configuration handling for llm_entropy package."""

import os
import yaml
import torch
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

@dataclass
class ControlConfig:
    enabled: bool = False
    layer_weights: Optional[List[float]] = None
    token_weights: Optional[List[float]] = None
    control_vectors: Optional[Dict[str, List[float]]] = None

    def to_tensors(self) -> Dict[str, torch.Tensor]:
        """Convert config values to tensors and expand control vectors."""
        if not self.control_vectors:
            return {'control_vector': None, 'layer_weights': None, 'token_weights': None}

        # Expand control vector to match model hidden size (768)
        pos_vector = self.control_vectors['positive']
        hidden_size = 768  # GPT-2's hidden size
        
        if len(pos_vector) < hidden_size:
            # Repeat the pattern to fill the vector
            pos_vector = pos_vector * (hidden_size // len(pos_vector) + 1)
            pos_vector = pos_vector[:hidden_size]  # Trim to exact size

        return {
            'control_vector': torch.tensor(pos_vector),
            'layer_weights': torch.tensor(self.layer_weights) if self.layer_weights else None,
            'token_weights': torch.tensor(self.token_weights) if self.token_weights else None
        }

@dataclass
class AnalysisConfig:
    num_sentences: int = 10
    temperature: float = 1.0
    use_sampling: bool = False
    compare_control: bool = False  # Add compare_control field
    control: Optional[ControlConfig] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'AnalysisConfig':
        if 'control' in d:
            d['control'] = ControlConfig(**d['control'])
        return cls(**d)

@dataclass
class ModelConfig:
    name: str
    output_hidden_states: bool = True
    control_dimension: Optional[int] = None

@dataclass
class OutputConfig:
    plots_dir: str = "plots"
    results_dir: str = "results"
    control_vectors_dir: Optional[str] = None

@dataclass
class Config:
    # System settings
    batch_size: int = 8
    max_workers: int = 4
    model_timeout: int = 300
    max_retries: int = 3
    cache_dir: str = ".cache"
    output_dir: str = "outputs"
    
    # Component configurations
    analysis: AnalysisConfig = AnalysisConfig()
    models: Dict[str, ModelConfig] = None
    output: OutputConfig = OutputConfig()
    
    def __post_init__(self):
        if self.models is None:
            self.models = {
                'gpt2': ModelConfig(name='gpt2'),
                'bert': ModelConfig(name='google-bert/bert-base-uncased')
            }
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'Config':
        if not os.path.exists(config_path):
            return cls()
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Process nested configurations
        analysis_dict = config_dict.pop('analysis', {})
        models_dict = config_dict.pop('models', {})
        output_dict = config_dict.pop('output', {})
        
        # Create component configs
        analysis_config = AnalysisConfig.from_dict(analysis_dict)
        models_config = {name: ModelConfig(**model_dict) 
                        for name, model_dict in models_dict.items()}
        output_config = OutputConfig(**output_dict)
        
        return cls(
            **config_dict,
            analysis=analysis_config,
            models=models_config,
            output=output_config
        )
    
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
            config = Config.from_yaml(config_path)
            
            # Convert control parameters to tensors if enabled
            if config.analysis.control and config.analysis.control.enabled:
                tensors = config.analysis.control.to_tensors()
                config.control_tensors = tensors
            
            return config
        except Exception as e:
            raise ValueError(f"Failed to load config from {config_path}: {str(e)}")
    
    # Return default config if no valid config file found
    return Config()
