#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.append('src')

from base_optimizer import BaseOptimizer

# Create a minimal test class that inherits from BaseOptimizer
class TestBounds(BaseOptimizer):
    def __init__(self):
        # Initialize with minimal parameters
        self.system_name = "CdS"
        self.base_param_file = "config/gfn1-base.toml"
        self.train_fraction = 0.8
        
        # Skip the full initialization, just set up what we need for bounds testing
        from config import get_system_config
        self.system_config = get_system_config(self.system_name)
        
    def optimize(self):
        pass  # Not needed for testing

# Test the bounds calculation
test_optimizer = TestBounds()

# Test sulfur slater parameters
sulfur_slater_tests = [
    ("element.S.slater[0]", 2.506934),
    ("element.S.slater[1]", 1.992775),
    ("element.S.slater[2]", 1.964867),
]

print("Testing fixed Slater bounds calculation:")
for param_name, default_val in sulfur_slater_tests:
    bounds = test_optimizer._get_parameter_bounds(param_name, default_val)
    print(f"  {param_name}:")
    print(f"    Default: {default_val}")
    print(f"    Bounds: {bounds}")
    if bounds[0] >= bounds[1]:
        print(f"    ERROR: Invalid bounds! min >= max")
    else:
        print(f"    ✓ Valid bounds")
    print() 
