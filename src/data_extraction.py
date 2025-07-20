"""
Parameter Extraction Module for GFN1-xTB Default Values
Extracts default parameters from TBLite/xTB for use in optimization algorithms
"""

import toml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Portable paths
PROJECT_ROOT = Path.cwd()
CONFIG_DIR = PROJECT_ROOT / "config"
BASE_PARAM_FILE = CONFIG_DIR / "gfn1-base.toml"

@dataclass
class ParameterInfo:
    """Information about a parameter for optimization"""
    name: str
    default_value: float
    suggested_min: float
    suggested_max: float
    description: str

class GFN1ParameterExtractor:
    """Extract default GFN1-xTB parameters for optimization"""
    
    def __init__(self, base_param_file: Optional[Path] = None):
        self.param_file = base_param_file or BASE_PARAM_FILE
        if not self.param_file.exists():
            raise FileNotFoundError(f"Base parameter file not found: {self.param_file}")
        
        with open(self.param_file, 'r') as f:
            self.params = toml.load(f)
    
    def extract_defaults_dict(self, elements: List[str] = ['H']) -> Dict[str, float]:
        """Extract default parameter values as simple dictionary"""
        defaults = {}
        
        # Hamiltonian parameters
        if 'hamiltonian' in self.params and 'xtb' in self.params['hamiltonian']:
            xtb = self.params['hamiltonian']['xtb']
            
            # Global parameters
            if 'kpol' in xtb:
                defaults['hamiltonian.xtb.kpol'] = xtb['kpol']
            if 'enscale' in xtb:
                defaults['hamiltonian.xtb.enscale'] = xtb['enscale']
            
            # Shell parameters
            if 'shell' in xtb:
                for param in ['ss', 'pp', 'sp']:
                    if param in xtb['shell']:
                        defaults[f'hamiltonian.xtb.shell.{param}'] = xtb['shell'][param]
            
            # Pair interactions
            if 'kpair' in xtb:
                for pair, value in xtb['kpair'].items():
                    defaults[f'hamiltonian.xtb.kpair.{pair}'] = value
        
        # Element parameters
        if 'element' in self.params:
            for element in elements:
                if element in self.params['element']:
                    elem = self.params['element'][element]
                    
                    # Array parameters
                    for array_name in ['levels', 'slater', 'kcn']:
                        if array_name in elem:
                            for i, value in enumerate(elem[array_name]):
                                defaults[f'element.{element}.{array_name}[{i}]'] = value
                    
                    # Single parameters
                    for param in ['gam', 'zeff', 'arep', 'en']:
                        if param in elem:
                            defaults[f'element.{element}.{param}'] = elem[param]
        
        return defaults

def extract_h2_parameters() -> Dict[str, float]:
    """Convenience function to extract H2-relevant parameters"""
    extractor = GFN1ParameterExtractor()
    return extractor.extract_defaults_dict(['H'])

def extract_si2_parameters() -> Dict[str, float]:
    """Extract only Si2-relevant parameters for focused optimization"""
    extractor = GFN1ParameterExtractor()
    defaults = {}
    
    # Si-Si pair interaction (most critical for Si2)
    if 'hamiltonian' in extractor.params and 'xtb' in extractor.params['hamiltonian']:
        xtb = extractor.params['hamiltonian']['xtb']
        if 'kpair' in xtb and 'Si-Si' in xtb['kpair']:
            defaults['hamiltonian.xtb.kpair.Si-Si'] = xtb['kpair']['Si-Si']
    
    # Key global parameters that affect Si bonding
    if 'hamiltonian' in extractor.params and 'xtb' in extractor.params['hamiltonian']:
        xtb = extractor.params['hamiltonian']['xtb']
        if 'kpol' in xtb:
            defaults['hamiltonian.xtb.kpol'] = xtb['kpol']
        if 'enscale' in xtb:
            defaults['hamiltonian.xtb.enscale'] = xtb['enscale']
    
    # Si element parameters (all critical for Si2)
    if 'element' in extractor.params and 'Si' in extractor.params['element']:
        elem = extractor.params['element']['Si']
        
        # Array parameters
        for array_name in ['levels', 'slater', 'kcn']:
            if array_name in elem:
                for i, value in enumerate(elem[array_name]):
                    defaults[f'element.Si.{array_name}[{i}]'] = value
        
        # Single parameters  
        for param in ['gam', 'zeff', 'arep', 'en']:
            if param in elem:
                defaults[f'element.Si.{param}'] = elem[param]
    
    return defaults

def extract_system_parameters(elements: List[str]) -> Dict[str, float]:
    """Extract only system-relevant parameters for focused optimization"""
    extractor = GFN1ParameterExtractor()
    defaults = {}
    
    # Generate all possible pair interactions for the system elements
    element_pairs = []
    for i, elem1 in enumerate(elements):
        for elem2 in elements[i:]:  # Include self-pairs and avoid duplicates
            element_pairs.append(f"{elem1}-{elem2}")
    
    # Extract pair interactions for the system elements
    if 'hamiltonian' in extractor.params and 'xtb' in extractor.params['hamiltonian']:
        xtb = extractor.params['hamiltonian']['xtb']
        if 'kpair' in xtb:
            for pair in element_pairs:
                if pair in xtb['kpair']:
                    defaults[f'hamiltonian.xtb.kpair.{pair}'] = xtb['kpair'][pair]
    
    # Key global parameters that affect bonding
    if 'hamiltonian' in extractor.params and 'xtb' in extractor.params['hamiltonian']:
        xtb = extractor.params['hamiltonian']['xtb']
        if 'kpol' in xtb:
            defaults['hamiltonian.xtb.kpol'] = xtb['kpol']
        if 'enscale' in xtb:
            defaults['hamiltonian.xtb.enscale'] = xtb['enscale']
        
        # Shell parameters
        if 'shell' in xtb:
            for param in ['ss', 'pp', 'sp']:
                if param in xtb['shell']:
                    defaults[f'hamiltonian.xtb.shell.{param}'] = xtb['shell'][param]
    
    # Element parameters for each element in the system
    for element in elements:
        if 'element' in extractor.params and element in extractor.params['element']:
            elem = extractor.params['element'][element]
            
            # Array parameters
            for array_name in ['levels', 'slater', 'kcn']:
                if array_name in elem:
                    for i, value in enumerate(elem[array_name]):
                        defaults[f'element.{element}.{array_name}[{i}]'] = value
            
            # Single parameters  
            for param in ['gam', 'zeff', 'arep', 'en']:
                if param in elem:
                    defaults[f'element.{element}.{param}'] = elem[param]
    
    return defaults

def extract_cds_parameters() -> Dict[str, float]:
    """Extract only CdS-relevant parameters for focused optimization"""
    return extract_system_parameters(['Cd', 'S'])

if __name__ == "__main__":
    print("GFN1-xTB Parameter Extraction")
    print("=" * 40)
    
    h2_defaults = extract_h2_parameters()
    print(f"Extracted {len(h2_defaults)} parameters:")
    for name, value in h2_defaults.items():
        print(f"  {name}: {value}") 
