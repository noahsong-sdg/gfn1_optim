
import os
import toml
import numpy as np
import matplotlib.pyplot
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse


def collect_and_plot_parameter_stats(
    toml_path='config/gfn1-base.toml',
    plot_dir='results/plots',
    bins=10
):
    """
    Reads a TOML file with element-wise parameters, computes statistics and histograms for each parameter
    across all elements, and saves histogram plots as PNGs.

    Args:
        toml_path (str): Path to the TOML file.
        plot_dir (str): Directory to save histogram plots.
        bins (int): Number of bins for histograms.

    Returns:
        dict: {parameter: {"values": [...], "min": float, "max": float, "mean": float, "std": float,
                          "hist": (counts, bin_edges), "plot_path": str}}
    """
    # Ensure plot directory exists
    os.makedirs(plot_dir, exist_ok=True)

    # Load TOML
    with open(toml_path, 'r') as f:
        data = toml.load(f)

    # --- FIX: Use nested structure for [element.X] tables ---
    if 'element' not in data:
        print("No 'element' section found in TOML file.")
        return {}
    element_sections = data['element']

    # Collect all parameter values across all elements
    param_values = {}
    for elem, params in element_sections.items():
        for param, value in params.items():
            # Flatten arrays, treat scalars as 1-element arrays
            if isinstance(value, list):
                vals = value
            else:
                vals = [value]
            # Skip if any value is not numeric
            if not all(isinstance(v, (int, float)) for v in vals):
                continue
            param_values.setdefault(param, []).extend(vals)

    # Compute stats and histograms
    stats = {}
    for param, values in param_values.items():
        arr = np.array(values, dtype=float)
        param_min = float(np.min(arr))
        param_max = float(np.max(arr))
        param_mean = float(np.mean(arr))
        param_std = float(np.std(arr))
        counts, bin_edges = np.histogram(arr, bins=bins)
        # Plot histogram
        plt.figure()
        plt.hist(arr, bins=bin_edges, edgecolor='black')
        plt.title(f'Histogram for {param}')
        plt.xlabel(param)
        plt.ylabel('Count')
        # Sanitize filename
        safe_param = ''.join(c if c.isalnum() or c in ('_', '-') else '_' for c in param)
        plot_path = os.path.join(plot_dir, f'{safe_param}.png')
        plt.savefig(plot_path)
        plt.close()
        # Store results
        stats[param] = {
            'values': arr.tolist(),
            'min': param_min,
            'max': param_max,
            'mean': param_mean,
            'std': param_std,
            'hist': (counts.tolist(), bin_edges.tolist()),
            'plot_path': plot_path
        }
    return stats

# DEBUG: Example usage (remove or comment out in production)
# stats = collect_and_plot_parameter_stats()
# for param, info in stats.items():
#     print(f"{param}: min={info['min']}, max={info['max']}, mean={info['mean']}, std={info['std']}")
#     print(f"  Histogram saved to: {info['plot_path']}")

def main():
    parser = argparse.ArgumentParser(description="Collect and plot parameter statistics from a TOML file.")
    parser.add_argument('--toml', type=str, default='config/gfn1-base.toml', help='Path to the TOML file')
    parser.add_argument('--plot_dir', type=str, default='results/plots', help='Directory to save histogram plots')
    parser.add_argument('--bins', type=int, default=10, help='Number of bins for histograms')
    args = parser.parse_args()

    stats = collect_and_plot_parameter_stats(
        toml_path=args.toml,
        plot_dir=args.plot_dir,
        bins=args.bins
    )
    print(f"Parameter statistics and histograms saved to {args.plot_dir}:")
    for param, info in stats.items():
        print(f"\nParameter: {param}")
        print(f"  min:  {info['min']}")
        print(f"  max:  {info['max']}")
        print(f"  mean: {info['mean']}")
        print(f"  std:  {info['std']}")
        print(f"  Histogram plot: {info['plot_path']}")

if __name__ == "__main__":
    main()
