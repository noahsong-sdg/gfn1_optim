"""
TBLite Parameter Optimization Package

A comprehensive package for optimizing TBLite parameters using various
optimization algorithms including Genetic Algorithm, Particle Swarm Optimization,
Bayesian Optimization, and CMA-ES.
"""

from .base_optimizer import BaseOptimizer, BaseConfig
from .config import get_system_config, SystemConfig, CalculationType
from .common import setup_logging, PROJECT_ROOT, CONFIG_DIR, RESULTS_DIR, DATA_DIR

# Import from subpackages
from .optimizers import *
from .calculators import *
from .utils import *
from .tests import *

__version__ = "1.0.0"
__author__ = "TBLite Optimization Team"

__all__ = [
    # Core classes
    'BaseOptimizer', 'BaseConfig',
    
    # Configuration
    'get_system_config', 'SystemConfig', 'CalculationType',
    
    # Common utilities
    'setup_logging', 'PROJECT_ROOT', 'CONFIG_DIR', 'RESULTS_DIR', 'DATA_DIR',
    
    # All optimizer classes (from optimizers package)
    'GeneralParameterGA', 'GAConfig',
    'GeneralParameterPSO', 'PSOConfig',
    'GeneralParameterBayesian', 'BayesianConfig',
    'GeneralParameterCMA', 'CMAConfig',
    'GeneralParameterCMA2', 'CMA2Config',
    
    # All calculator classes (from calculators package)
    'GeneralCalculator', 'DissociationCurveGenerator', 'CrystalGenerator',
    'CalcConfig', 'CalcMethod', 'TBLiteASECalculator',
    
    # All utility classes (from utils package)
    'GFN1ParameterExtractor', 'extract_system_parameters',
    'ParameterBoundsManager', 'ParameterBounds',
    
    # All test classes (from tests package)
    'MethodComparisonAnalyzer', 'BandgapCalculator', 'BulkMaterialsValidator'
] 
