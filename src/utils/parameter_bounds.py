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
    ENERGY_LEVEL = "energy_level"           # Atomic self-energies (array of reals)
    SLATER_EXPONENT = "slater_exponent"     # slater exponents of basis functions (array of reals)
    E_SHIFT = "e_shift"           # CN dependent self-energy shift
    SHELL_HARDNESS = "shell_hardness"   # Pair interaction parameters (array of reals)
    GAMMA = "gamma"                         # Gamma parameters (0.1-1.0)
    SHELL_PARAMETER = "shell_parameter"     # Shell parameters (positive)
    XBOND = "xbond"                         # Bonding parameters (0.0-0.05)
    GAMMA_DERIVATIVE = "gamma_derivative"   # Gamma derivative parameters (0.01-0.2)
    PAIR = "pair"                           # Pair interaction parameters (0.5-1.5)
    EFFECTIVE_CHARGE = "effective_charge"   # Effective nuclear charge (positive)
    REPULSION = "repulsion"                 # Repulsion parameters (positive)
    ELECTRONEGATIVITY = "electronegativity" # Electronegativity (positive)
    POLARIZATION = "polarization"           # Polarization parameters (2.5-3.5)
    ENERGY_SCALE = "energy_scale"           # Energy scaling (-0.015 to 0.002)
    POLY_PARAMETER = "poly_parameter"     # Shell parameters (positive)
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

# not optimizing:
    # refocc (reference occupation of atom)
    # zeff 
    # dkernel everything is zero
    # same for qkernel and mpvcn and mprad

class ParameterBoundsManager:
    """
    Centralized parameter bounds management system.
    """
    PARAMETER_CONSTRAINTS = {
        # levels array
        ParameterType.ENERGY_LEVEL: ParameterConstraint(
            param_type=ParameterType.ENERGY_LEVEL,
            min_val= -40.0,  # min: -31.395426999999
            max_val= 2.0,   # max: 1.487984
            description=" negative)",
            physical_justification="Energy levels represent atomic orbital energies, which are typically negative for bound states"
        ),
        # slater array
        ParameterType.SLATER_EXPONENT: ParameterConstraint(
            param_type=ParameterType.SLATER_EXPONENT,
            min_val= 0.1,    # min: 0.532658
            max_val= 5.0,    # 3.520683
            description="Slater exponents (positive)",
            physical_justification="Slater exponents control orbital decay and must be positive"
        ),
        # shpoly array
        ParameterType.POLY_PARAMETER: ParameterConstraint(
            param_type=ParameterType.POLY_PARAMETER,
            min_val= -0.5,    # min: -0.44208463
            max_val= 0.5,    # max: 0.4567976
            description="Poly parameters (positive)",
            physical_justification="Poly parameters must be positive for physical meaning"
        ),
        # kcn array
        ParameterType.E_SHIFT: ParameterConstraint(
            param_type=ParameterType.E_SHIFT,
            min_val=-0.1,   
            max_val=0.1,   # Reasonable upper limit
            description="CN dependent self-energy shift",
            physical_justification="CN dependent self-energy shift parameters must be positive for physical meaning"
        ),
        # gam
        ParameterType.GAMMA: ParameterConstraint(
            param_type=ParameterType.GAMMA,
            min_val= 0.01,    # min: 0.075735
            max_val= 2.0,    # max: 1.441379
            description="Gamma parameters (0.01-2.0)",
            physical_justification="Gamma parameters control orbital overlap and should be in reasonable range"
        ),
        # lgam
        ParameterType.SHELL_HARDNESS: ParameterConstraint(
            param_type=ParameterType.SHELL_HARDNESS,
            min_val= 0.01, #0.1198637
            max_val=2.5, # 2.1522018
            description="Shell hardness parameters (0.01-2.5)",
            physical_justification="Shell hardness parameters represent the hardness of the shell and must be positive"
        ),
        # gam3 
        ParameterType.GAMMA_DERIVATIVE: ParameterConstraint(
            param_type=ParameterType.GAMMA_DERIVATIVE,
            min_val= -0.01,    # min: -0.0860502
            max_val= 0.25,    # max: 0.1615037
            description="Gamma derivative parameters (0.01-0.2)",
            physical_justification="Gamma derivative parameters control orbital overlap and should be in reasonable range"
        ),
        # arep
        ParameterType.REPULSION: ParameterConstraint(
            param_type=ParameterType.REPULSION,
            min_val= 0.1,    # min: 0.554032
            max_val= 4.0,    # max: 3.038727
            description="Repulsion parameters (0.1-4.0)",
            physical_justification="Repulsion parameters represent the repulsion between atoms and must be positive"
        ),
        # xbond
        ParameterType.XBOND: ParameterConstraint(
            param_type=ParameterType.XBOND,
            min_val= 0.0,    # min: 0.0
            max_val= 0.05,    # max: 0.0381742
            description="Bonding parameters (0.0-0.05)",
            physical_justification="Bonding parameters represent the bonding strength and must be positive"
        ),
        # en
        ParameterType.ELECTRONEGATIVITY: ParameterConstraint(
            param_type=ParameterType.ELECTRONEGATIVITY,
            min_val= 0.1,    # min: 0.79
            max_val= 5.5,    # max: 4.5
            description="Electronegativity parameters (0.1-4.0)",
            physical_justification="Electronegativity parameters represent the electronegativity of the atom and must be positive"
        ),
        # kpair
        ParameterType.PAIR: ParameterConstraint(
            param_type=ParameterType.PAIR,
            min_val=0.5,    
            max_val=1.5,    
            description="Pair interaction parameters",
            physical_justification="Pair interactions represent bonding strength and must be positive"
        ),      
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
            param_type = ParameterType.ENERGY_LEVEL # levels
        elif 'slater[' in param_name:
            param_type = ParameterType.SLATER_EXPONENT #slater
        elif 'shpoly[' in param_name:
            param_type = ParameterType.POLY_PARAMETER #shpoly
        elif 'kcn[' in param_name:
            param_type = ParameterType.E_SHIFT #kcn
        elif 'kpair' in param_name:
            param_type = ParameterType.PAIR #kpair
        elif 'gam' in param_name and not param_name.endswith('lgam'):
            param_type = ParameterType.GAMMA #gam
        elif 'lgam' in param_name:
            param_type = ParameterType.SHELL_HARDNESS #lgam
        elif 'gam3' in param_name:
            param_type = ParameterType.GAMMA_DERIVATIVE #gam3
        elif 'arep' in param_name:
            param_type = ParameterType.REPULSION #arep
        elif 'en' in param_name and not param_name.endswith('zen'):
            param_type = ParameterType.ELECTRONEGATIVITY #en
        elif 'kpol' in param_name:
            param_type = ParameterType.POLARIZATION #kpol
        elif 'enscale' in param_name:
            param_type = ParameterType.ENERGY_SCALE #enscale
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
    
    def create_static_parameter_bounds(self, param_name: str, default_val: float) -> ParameterBounds:
        """
        Create ParameterBounds using only static min/max from PARAMETER_CONSTRAINTS.
        Change this method if you want dynamic bounds in the future.
        """
        constraint = self.get_parameter_constraint(param_name)
        return ParameterBounds(
            name=param_name,
            min_val=constraint.min_val,
            max_val=constraint.max_val,
            default_val=default_val,
            param_type=constraint.param_type,
            description=constraint.description
        )

    # DEPRECATED: Use create_static_parameter_bounds instead if you want static bounds only.
    def create_parameter_bounds(self, param_name: str, default_val: float, margin_factor: float = 0.15) -> ParameterBounds:
        """
        DEPRECATED: This method uses dynamic bounds. Use create_static_parameter_bounds for static bounds.
        """
        return self.create_static_parameter_bounds(param_name, default_val)
    
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


def create_10p_parameter_bounds(defaults: Dict[str, Any]) -> List[ParameterBounds]:
    """
    Create ParameterBounds for each parameter with bounds set to ±10% of the default value,
    but not exceeding the absolute min/max for that parameter type as defined in ParameterBoundsManager.
    Handles both scalar and array-valued parameters.

    Args:
        defaults: Dict of parameter names to their default values (float or list of floats).
    Returns:
        List[ParameterBounds]: List of ParameterBounds objects with 10% bounds, clamped to absolute limits.
    """
    manager = ParameterBoundsManager()
    bounds_list = []
    for param_name, default_val in defaults.items():
        # Handle array-valued parameters
        if isinstance(default_val, (list, tuple, np.ndarray)):
            for i, v in enumerate(default_val):
                constraint = manager.get_parameter_constraint(f"{param_name}[{i}]")
                margin = abs(v) * 0.10
                min_val = max(constraint.min_val, v - margin)
                max_val = min(constraint.max_val, v + margin)
                if min_val >= max_val:
                    min_val = constraint.min_val
                    max_val = constraint.max_val
                bounds = ParameterBounds(
                    name=f"{param_name}[{i}]",
                    min_val=min_val,
                    max_val=max_val,
                    default_val=v,
                    param_type=constraint.param_type,
                    description=constraint.description
                )
                bounds_list.append(bounds)
        else:
            constraint = manager.get_parameter_constraint(param_name)
            margin = abs(default_val) * 0.10
            min_val = max(constraint.min_val, default_val - margin)
            max_val = min(constraint.max_val, default_val + margin)
            if min_val >= max_val:
                min_val = constraint.min_val
                max_val = constraint.max_val
            bounds = ParameterBounds(
                name=param_name,
                min_val=min_val,
                max_val=max_val,
                default_val=default_val,
                param_type=constraint.param_type,
                description=constraint.description
            )
            bounds_list.append(bounds)
    return bounds_list 
