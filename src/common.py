"""
Common configuration and utilities for the TBLite optimization project.
Centralizes path constants, logging setup, and common imports.
"""

import logging
import os
from pathlib import Path
from typing import Optional

# Portable paths - centralized for all modules
PROJECT_ROOT = Path.cwd()
CONFIG_DIR = PROJECT_ROOT / "config"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"
RANDOM_SEED = 42

def setup_logging(level: str = "INFO", 
                 format_str: Optional[str] = None,
                 module_name: Optional[str] = None) -> logging.Logger:
    """Centralized logging setup for consistent configuration across modules"""
    
    if format_str is None:
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Only configure logging once
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format=format_str,
            datefmt='%H:%M:%S'
        )
    
    # Create module-specific logger
    if module_name:
        logger = logging.getLogger(module_name)
    else:
        logger = logging.getLogger(__name__)
    
    # Suppress verbose output from external libraries
    logging.getLogger('cmaes').setLevel(logging.WARNING)
    logging.getLogger('cma').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('numpy').setLevel(logging.WARNING)
    
    return logger

# Common imports that are used across multiple modules
COMMON_IMPORTS = {
    'numpy': 'np',
    'pandas': 'pd',
    'pathlib': 'Path',
    'logging': 'logging',
    'typing': ['Dict', 'List', 'Optional', 'Tuple', 'Any', 'Union'],
    'dataclasses': 'dataclass',
    'tempfile': 'tempfile',
    'copy': 'copy',
    'time': 'time',
    'pickle': 'pickle',
    'random': 'random',
    'os': 'os'
}

def get_common_imports() -> str:
    """Return common import statements as a string for consistency"""
    imports = []
    
    # Standard library imports
    imports.append("import numpy as np")
    imports.append("import pandas as pd")
    imports.append("import logging")
    imports.append("import os")
    imports.append("import copy")
    imports.append("import tempfile")
    imports.append("import time")
    imports.append("import pickle")
    imports.append("import random")
    imports.append("from pathlib import Path")
    imports.append("from typing import Dict, List, Optional, Tuple, Any, Union")
    imports.append("from dataclasses import dataclass")
    
    return "\n".join(imports)

# Default configuration values
DEFAULT_CONFIG = {
    'convergence_threshold': 1e-6,
    'patience': 20,
    'max_workers': 4,
    'train_fraction': 0.8,
    'random_seed': 42
} 
