#!/usr/bin/env python3

from pathlib import Path
from src.data_extraction import GFN1ParameterExtractor

# Extract parameters
extractor = GFN1ParameterExtractor(Path('config/gfn1-base.toml'))
params = extractor.extract_defaults_dict(['H', 'S'])

# Find sulfur slater parameters
sulfur_slater = {k: v for k, v in params.items() if 'slater' in k and 'S' in k}

print("Sulfur slater parameters:")
for param_name, value in sulfur_slater.items():
    print(f"  {param_name}: {value}")

# Also check what the bounds calculation would be
print("\nCalculating bounds for sulfur slater parameters:")
for param_name, default_val in sulfur_slater.items():
    min_val = max(0.5, default_val * 0.8)
    max_val = min(2.0, default_val * 1.2)
    print(f"  {param_name}:")
    print(f"    Default: {default_val}")
    print(f"    Min: {min_val} (max(0.5, {default_val} * 0.8))")
    print(f"    Max: {max_val} (min(2.0, {default_val} * 1.2))")
    print(f"    Bounds: ({min_val}, {max_val})")
    if min_val >= max_val:
        print(f"    ERROR: Invalid bounds! min >= max")
    print() 
