"""
Configuration loading and validation utilities.
"""

import yaml
import os
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
        
    except Exception as e:
        raise RuntimeError(f"Failed to load configuration: {e}")


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    required_sections = ['dataset', 'slam']
    
    # Check required sections
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate dataset section
    dataset_config = config['dataset']
    required_dataset_keys = ['type', 'camera']
    for key in required_dataset_keys:
        if key not in dataset_config:
            raise ValueError(f"Missing required dataset configuration key: {key}")
    
    # Validate SLAM section
    slam_config = config['slam']
    required_slam_keys = ['window_size', 'orb_features']
    for key in required_slam_keys:
        if key not in slam_config:
            raise ValueError(f"Missing required SLAM configuration key: {key}")
    
    # Validate ranges
    if slam_config['window_size'] < 2:
        raise ValueError("Window size must be at least 2")
    
    if slam_config['orb_features'] < 100:
        raise ValueError("Number of ORB features must be at least 100")
    
    logger.debug("Configuration validation passed")
    return True


def get_default_config() -> Dict[str, Any]:
    """Get default configuration."""
    return {
        'dataset': {
            'type': 'euroc',
            'camera': 'cam0',
            'downsample_factor': 10,
        },
        'slam': {
            'window_size': 5,
            'orb_features': 1000,
            'loop_closure': {
                'enabled': True,
                'vocabulary_size': 500,
                'similarity_threshold': 0.7,
            },
            'optimization': {
                'max_iterations': 100,
                'verbose': False,
            },
        },
        'visualization': {
            'show_features': True,
            'show_trajectory': True,
            'save_plots': True,
        },
    }


def save_config(config: Dict[str, Any], save_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save configuration file
    """
    try:
        with open(save_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Saved configuration to {save_path}")
        
    except Exception as e:
        logger.error(f"Failed to save configuration: {e}")
        raise