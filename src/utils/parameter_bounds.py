"""
Centralized Parameter Bounds Management System

This module provides a comprehensive, scientific approach to managing parameter bounds
for GFN1-xTB parameter optimization. It replaces the scattered bounds logic across
multiple files with a single, robust system.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)

class ParameterType(Enum):
    """Enumeration of parameter types for scientific bounds management"""
    ENERGY_LEVEL = "energy_level"           # Atomic energy levels (can be negative)
    SLATER_EXPONENT = "slater_exponent"     # Slater exponents (positive)
    COORDINATION = "coordination"           # Coordination numbers (positive)
    PAIR_INTERACTION = "pair_interaction"   # Pair interaction parameters (positive)
    GAMMA = "gamma"                         # Gamma parameters (0.1-1.0)
    EFFECTIVE_CHARGE = "effective_charge"   # Effective nuclear charge (positive)
    REPULSION = "repulsion"                 # Repulsion parameters (positive)
    ELECTRONEGATIVITY = "electronegativity" # Electronegativity (positive)
    POLARIZATION = "polarization"           # Polarization parameters (2.5-3.5)
    ENERGY_SCALE = "energy_scale"           # Energy scaling (-0.015 to 0.002)
    SHELL_PARAMETER = "shell_parameter"     # Shell parameters (positive)
    UNKNOWN = "unknown"                     # Unknown parameter type

@dataclass
class ParameterConstraint:
    """Scientific constraints for a parameter type"""
    param_type: ParameterType
    min_val: float
    max_val: float
    description: str
    physical_justification: str

@dataclass
class ParameterBounds:
    """Comprehensive parameter bounds with scientific validation"""
    name: str
    min_val: float
    max_val: float
    default_val: float
    param_type: ParameterType
    description: str
    
    def __post_init__(self):
        """Validate bounds after initialization"""
        if self.min_val >= self.max_val:
            raise ValueError(f"Invalid bounds for {self.name}: min={self.min_val} >= max={self.max_val}")
        
        if not (self.min_val <= self.default_val <= self.max_val):
            raise ValueError(f"Default value {self.default_val} for {self.name} not within bounds [{self.min_val}, {self.max_val}]")

class ParameterBoundsManager:
    """
    Centralized parameter bounds management system.
    
    This class provides a scientific, consistent approach to managing parameter bounds
    across all optimizers. It replaces the scattered bounds logic with a single
    source of truth.
    """
    
    # Scientific parameter constraints based on physical principles
    PARAMETER_CONSTRAINTS = {
        ParameterType.ENERGY_LEVEL: ParameterConstraint(
            param_type=ParameterType.ENERGY_LEVEL,
            min_val=-50.0,  # Energy levels can be significantly negative
            max_val=10.0,   # But not extremely positive
            description="Atomic energy levels (can be negative)",
            physical_justification="Energy levels represent atomic orbital energies, which are typically negative for bound states"
        ),
        ParameterType.SLATER_EXPONENT: ParameterConstraint(
            param_type=ParameterType.SLATER_EXPONENT,
            min_val=0.1,    # Must be positive for physical meaning
            max_val=5.0,    # Reasonable upper limit
            description="Slater exponents (positive)",
            physical_justification="Slater exponents control orbital decay and must be positive"
        ),
        ParameterType.COORDINATION: ParameterConstraint(
            param_type=ParameterType.COORDINATION,
            min_val=0.01,   # Small positive value
            max_val=10.0,   # Reasonable upper limit
            description="Coordination numbers (positive)",
            physical_justification="Coordination parameters must be positive for physical meaning"
        ),
        ParameterType.PAIR_INTERACTION: ParameterConstraint(
            param_type=ParameterType.PAIR_INTERACTION,
            min_val=0.1,    # Must be positive
            max_val=2.0,    # Reasonable upper limit
            description="Pair interaction parameters (positive)",
            physical_justification="Pair interactions represent bonding strength and must be positive"
        ),
        ParameterType.GAMMA: ParameterConstraint(
            param_type=ParameterType.GAMMA,
            min_val=0.1,    # Lower bound for stability
            max_val=1.0,    # Upper bound for reasonable values
            description="Gamma parameters (0.1-1.0)",
            physical_justification="Gamma parameters control orbital overlap and should be in reasonable range"
        ),
        ParameterType.EFFECTIVE_CHARGE: ParameterConstraint(
            param_type=ParameterType.EFFECTIVE_CHARGE,
            min_val=1.0,    # Must be at least 1 for any element
            max_val=100.0,  # Upper limit for heavy elements
            description="Effective nuclear charge (positive)",
            physical_justification="Effective nuclear charge must be positive and typically >= 1"
        ),
        ParameterType.REPULSION: ParameterConstraint(
            param_type=ParameterType.REPULSION,
            min_val=0.1,    # Must be positive
            max_val=2.0,    # Reasonable upper limit
            description="Repulsion parameters (positive)",
            physical_justification="Repulsion parameters must be positive for physical meaning"
        ),
        ParameterType.ELECTRONEGATIVITY: ParameterConstraint(
            param_type=ParameterType.ELECTRONEGATIVITY,
            min_val=0.1,    # Must be positive
            max_val=5.0,    # Reasonable upper limit
            description="Electronegativity (positive)",
            physical_justification="Electronegativity must be positive for physical meaning"
        ),
        ParameterType.POLARIZATION: ParameterConstraint(
            param_type=ParameterType.POLARIZATION,
            min_val=2.5,    # Lower bound for stability
            max_val=3.5,    # Upper bound for stability
            description="Polarization parameters (2.5-3.5)",
            physical_justification="Polarization parameters have narrow range for numerical stability"
        ),
        ParameterType.ENERGY_SCALE: ParameterConstraint(
            param_type=ParameterType.ENERGY_SCALE,
            min_val=-0.015, # Lower bound for stability
            max_val=0.002,  # Upper bound for stability
            description="Energy scaling (-0.015 to 0.002)",
            physical_justification="Energy scaling has narrow range for numerical stability"
        ),
        ParameterType.SHELL_PARAMETER: ParameterConstraint(
            param_type=ParameterType.SHELL_PARAMETER,
            min_val=0.1,    # Must be positive
            max_val=2.0,    # Reasonable upper limit
            description="Shell parameters (positive)",
            physical_justification="Shell parameters must be positive for physical meaning"
        )
    }
    
    def __init__(self):
        """Initialize the bounds manager"""
        self._parameter_type_cache: Dict[str, ParameterType] = {}
    
    def classify_parameter(self, param_name: str) -> ParameterType:
        """
        Classify a parameter based on its name using scientific rules.
        
        Args:
            param_name: Parameter name (e.g., 'element.Cd.levels[0]')
            
        Returns:
            ParameterType: The classified parameter type
        """
        # Check cache first
        if param_name in self._parameter_type_cache:
            return self._parameter_type_cache[param_name]
        
        # Classification rules based on parameter name patterns
        if 'levels[' in param_name:
            param_type = ParameterType.ENERGY_LEVEL
        elif 'slater[' in param_name:
            param_type = ParameterType.SLATER_EXPONENT
        elif 'kcn[' in param_name:
            param_type = ParameterType.COORDINATION
        elif 'kpair' in param_name:
            param_type = ParameterType.PAIR_INTERACTION
        elif 'gam' in param_name and not param_name.endswith('lgam'):
            param_type = ParameterType.GAMMA
        elif 'zeff' in param_name:
            param_type = ParameterType.EFFECTIVE_CHARGE
        elif 'arep' in param_name:
            param_type = ParameterType.REPULSION
        elif 'en' in param_name and not param_name.endswith('zen'):
            param_type = ParameterType.ELECTRONEGATIVITY
        elif 'kpol' in param_name:
            param_type = ParameterType.POLARIZATION
        elif 'enscale' in param_name:
            param_type = ParameterType.ENERGY_SCALE
        elif any(shell in param_name for shell in ['ss', 'pp', 'sp']):
            param_type = ParameterType.SHELL_PARAMETER
        else:
            param_type = ParameterType.UNKNOWN
        
        # Cache the result
        self._parameter_type_cache[param_name] = param_type
        return param_type
    
    def get_parameter_constraint(self, param_name: str) -> ParameterConstraint:
        """
        Get the scientific constraint for a parameter.
        
        Args:
            param_name: Parameter name
            
        Returns:
            ParameterConstraint: The constraint for this parameter type
        """
        param_type = self.classify_parameter(param_name)
        return self.PARAMETER_CONSTRAINTS.get(param_type, ParameterConstraint(
            param_type=ParameterType.UNKNOWN,
            min_val=-10.0,  # Conservative default
            max_val=10.0,   # Conservative default
            description="Unknown parameter type",
            physical_justification="No specific physical constraints known"
        ))
    
    def calculate_bounds(self, param_name: str, default_val: float, 
                        margin_factor: float = 0.15) -> Tuple[float, float]:
        """
        Calculate scientifically justified bounds for a parameter.
        
        Args:
            param_name: Parameter name
            default_val: Default value from TOML file
            margin_factor: Factor for calculating bounds around default (default: 0.15)
            
        Returns:
            Tuple[float, float]: (min_val, max_val)
        """
        constraint = self.get_parameter_constraint(param_name)
        
        # Calculate bounds based on default value and constraint
        if constraint.param_type == ParameterType.ENERGY_LEVEL:
            # Energy levels: allow negative values, use asymmetric bounds
            if default_val < 0:
                # For negative energy levels, allow more variation downward
                min_val = default_val * 1.5  # Allow 50% more negative
                max_val = default_val * 0.5  # Allow 50% less negative
            else:
                # For positive energy levels, use symmetric bounds
                margin = abs(default_val) * margin_factor
                min_val = max(constraint.min_val, default_val - margin)
                max_val = min(constraint.max_val, default_val + margin)
        
        elif constraint.param_type in [ParameterType.POLARIZATION, ParameterType.ENERGY_SCALE]:
            # Use fixed bounds for stability-critical parameters
            min_val = constraint.min_val
            max_val = constraint.max_val
        
        else:
            # For other parameters, use constraint bounds with margin
            margin = abs(default_val) * margin_factor
            min_val = max(constraint.min_val, default_val - margin)
            max_val = min(constraint.max_val, default_val + margin)
        
        # Final validation
        if min_val >= max_val:
            # Fallback to constraint bounds if calculated bounds are invalid
            min_val = constraint.min_val
            max_val = constraint.max_val
            logger.warning(f"Invalid calculated bounds for {param_name}, using constraint bounds: [{min_val}, {max_val}]")
        
        return min_val, max_val
    
    def create_parameter_bounds(self, param_name: str, default_val: float, 
                               margin_factor: float = 0.15) -> ParameterBounds:
        """
        Create a ParameterBounds object with scientific validation.
        
        Args:
            param_name: Parameter name
            default_val: Default value
            margin_factor: Factor for calculating bounds
            
        Returns:
            ParameterBounds: The parameter bounds object
        """
        param_type = self.classify_parameter(param_name)
        constraint = self.get_parameter_constraint(param_name)
        min_val, max_val = self.calculate_bounds(param_name, default_val, margin_factor)
        
        return ParameterBounds(
            name=param_name,
            min_val=min_val,
            max_val=max_val,
            default_val=default_val,
            param_type=param_type,
            description=constraint.description
        )
    
    def validate_parameters(self, parameters: Dict[str, float], 
                          bounds: List[ParameterBounds]) -> List[str]:
        """
        Validate that parameters are within bounds.
        
        Args:
            parameters: Parameter dictionary
            bounds: List of parameter bounds
            
        Returns:
            List[str]: List of validation errors (empty if valid)
        """
        errors = []
        bounds_dict = {b.name: b for b in bounds}
        
        for param_name, value in parameters.items():
            if param_name in bounds_dict:
                bound = bounds_dict[param_name]
                if value < bound.min_val or value > bound.max_val:
                    errors.append(f"Parameter {param_name}={value} outside bounds [{bound.min_val}, {bound.max_val}]")
        
        return errors
    
    def apply_bounds(self, parameters: Dict[str, float], 
                    bounds: List[ParameterBounds]) -> Dict[str, float]:
        """
        Apply bounds by clamping parameter values.
        
        Args:
            parameters: Parameter dictionary
            bounds: List of parameter bounds
            
        Returns:
            Dict[str, float]: Bounded parameters
        """
        bounded_params = {}
        bounds_dict = {b.name: b for b in bounds}
        
        for param_name, value in parameters.items():
            if param_name in bounds_dict:
                bound = bounds_dict[param_name]
                bounded_value = max(bound.min_val, min(bound.max_val, value))
                bounded_params[param_name] = float(bounded_value)
            else:
                bounded_params[param_name] = float(value)
        
        return bounded_params
    
    def get_bounds_summary(self, bounds: List[ParameterBounds]) -> Dict[str, Any]:
        """
        Get a summary of parameter bounds for logging and analysis.
        
        Args:
            bounds: List of parameter bounds
            
        Returns:
            Dict[str, Any]: Summary statistics
        """
        if not bounds:
            return {"error": "No bounds provided"}
        
        # Group by parameter type
        type_counts = {}
        type_ranges = {}
        
        for bound in bounds:
            param_type = bound.param_type.value
            if param_type not in type_counts:
                type_counts[param_type] = 0
                type_ranges[param_type] = []
            
            type_counts[param_type] += 1
            type_ranges[param_type].append(bound.max_val - bound.min_val)
        
        # Calculate statistics
        summary = {
            "total_parameters": len(bounds),
            "parameter_types": type_counts,
            "average_ranges": {pt: np.mean(ranges) for pt, ranges in type_ranges.items()},
            "min_ranges": {pt: np.min(ranges) for pt, ranges in type_ranges.items()},
            "max_ranges": {pt: np.max(ranges) for pt, ranges in type_ranges.items()}
        }
        
        return summary
    
    def log_bounds_summary(self, bounds: List[ParameterBounds], system_name: str):
        """
        Log a comprehensive summary of parameter bounds.
        
        Args:
            bounds: List of parameter bounds
            system_name: Name of the system being optimized
        """
        summary = self.get_bounds_summary(bounds)
        
        logger.info(f"Parameter bounds summary for {system_name}:")
        logger.info(f"  Total parameters: {summary['total_parameters']}")
        
        for param_type, count in summary['parameter_types'].items():
            avg_range = summary['average_ranges'][param_type]
            min_range = summary['min_ranges'][param_type]
            max_range = summary['max_ranges'][param_type]
            logger.info(f"  {param_type}: {count} parameters, range: [{min_range:.4f}, {max_range:.4f}], avg: {avg_range:.4f}")
        
        # Log examples of each parameter type
        logger.info("Parameter examples by type:")
        type_examples = {}
        for bound in bounds:
            param_type = bound.param_type.value
            if param_type not in type_examples:
                type_examples[param_type] = []
            if len(type_examples[param_type]) < 3:  # Show up to 3 examples
                type_examples[param_type].append(f"{bound.name}: [{bound.min_val:.4f}, {bound.max_val:.4f}]")
        
        for param_type, examples in type_examples.items():
            logger.info(f"  {param_type}: {', '.join(examples)}") 
