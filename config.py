"""
Configuration file for energy analysis.
"""
from pathlib import Path
from typing import Dict, List, Any
import json

# Default configuration
DEFAULT_CONFIG = {
    # Device classification
    "device_classes": {
        "DB16": [
            "0xa4c138e2531ae4a2",
            "0xa4c138b04dad49bf",
            "0xa4c138309b3a04fe",
            "0xa4c13808db35efce",
  
        ],
        "DB18": [
            "0xa4c138e8428b8966",
            "0xa4c1384306f7e9b8",    
        ],
        "DB14": [
            "0xa4c138344844cee8",
            "0xa4c138a67215d121",
            "0xa4c13803d6ce7cb2",
        ]
    },
    
    # Analysis window
    "analysis_window": {
        "start": "2025-02-15T00:00:00",
        "end": "2025-05-20T23:59:59",
        "timezone": "Europe/Istanbul"
    },
    
    # Visualization settings
    "visualization": {
        "output_dir": "visualizations",
        "figure_size": (12, 8),
        "font_size": 12,
        "color_palette": [
            "#0072B2", "#D55E00", "#009E73", 
            "#CC79A7", "#F0E442", "#56B4E9", "#E69F00"
        ]
    },
    
    # Light codes for state analysis
    "light_codes": ["state", "state_l1", "state_l2"],
    
    # Energy analysis settings
    "energy_analysis": {
        "resample_method": "ffill",  # forward fill missing values
        "daily_aggregation": "sum",   # sum, mean
        "hourly_aggregation": "mean", # sum, mean
        "filter_negative_consumption": True,
        "min_valid_readings": 5,      # minimum number of readings for a device to be included
        "gap_threshold_hours": 24     # threshold (in hours) for identifying significant gaps in data transmission
    },
    "anomaly_detection": {
        "jump_threshold_factor": 2.0,  # factor above expected consumption to identify abnormal jumps
        "iqr_multiplier": 1.5,         # multiplier for IQR-based outlier detection
        "use_time_weighted_distribution": False,  # whether to use time-weighted distribution across gaps
        "log_anomalies": True          # whether to log details about detected anomalies
    }
}

def load_config(config_path: str = "energy_config.json") -> Dict[str, Any]:
    """
    Load configuration from file or return default if file doesn't exist.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            # Merge with defaults to ensure all required keys exist
            merged_config = DEFAULT_CONFIG.copy()
            _deep_update(merged_config, config)
            return merged_config
    except (FileNotFoundError, json.JSONDecodeError):
        return DEFAULT_CONFIG.copy()

def save_config(config: Dict[str, Any], config_path: str = "energy_config.json") -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration file
    """
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

def _deep_update(source: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update a nested dictionary with another nested dictionary.
    
    Args:
        source: Source dictionary to be updated
        update: Dictionary with updates
        
    Returns:
        Updated source dictionary
    """
    for key, value in update.items():
        if key in source and isinstance(source[key], dict) and isinstance(value, dict):
            _deep_update(source[key], value)
        else:
            source[key] = value
    return source    