"""
Configuration module for sentiment-controlled text generation.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import yaml
import json


@dataclass
class ModelConfig:
    """Configuration for model settings."""
    name: str = "gpt2"
    device: str = "auto"
    use_accelerate: bool = True
    torch_dtype: str = "auto"


@dataclass
class GenerationConfig:
    """Configuration for text generation parameters."""
    max_length: int = 100
    num_return_sequences: int = 1
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.1


@dataclass
class UIConfig:
    """Configuration for user interface settings."""
    title: str = "Sentiment-Controlled Text Generation"
    description: str = "Generate text with controlled sentiment using AI"
    max_prompt_length: int = 500
    default_num_samples: int = 3


@dataclass
class AppConfig:
    """Main application configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    logging_level: str = "INFO"
    cache_dir: str = "./cache"
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'AppConfig':
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, config_path: str) -> 'AppConfig':
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def to_yaml(self, config_path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            'model': {
                'name': self.model.name,
                'device': self.model.device,
                'use_accelerate': self.model.use_accelerate,
                'torch_dtype': self.model.torch_dtype
            },
            'generation': {
                'max_length': self.generation.max_length,
                'num_return_sequences': self.generation.num_return_sequences,
                'temperature': self.generation.temperature,
                'top_p': self.generation.top_p,
                'top_k': self.generation.top_k,
                'do_sample': self.generation.do_sample,
                'repetition_penalty': self.generation.repetition_penalty
            },
            'ui': {
                'title': self.ui.title,
                'description': self.ui.description,
                'max_prompt_length': self.ui.max_prompt_length,
                'default_num_samples': self.ui.default_num_samples
            },
            'logging_level': self.logging_level,
            'cache_dir': self.cache_dir
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)


def get_default_config() -> AppConfig:
    """Get default configuration."""
    return AppConfig()


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Load configuration from file or return default."""
    if config_path and os.path.exists(config_path):
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            return AppConfig.from_yaml(config_path)
        elif config_path.endswith('.json'):
            return AppConfig.from_json(config_path)
    
    return get_default_config()
