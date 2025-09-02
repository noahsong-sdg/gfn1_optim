import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def plot_generation_vs_best_fitness(csv_path, save_path=None):
    """
    Plot the 'generation' vs 'best_fitness' from a CSV file.

    Args:
        csv_path (str): Path to the CSV file containing columns 'generation' and 'best_fitness'.
        save_path (str, optional): Path to save the plot. If None, auto-generates from CSV path.
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Check required columns
    if 'generation' not in df.columns or 'best_fitness' not in df.columns:
        raise ValueError("CSV must contain 'generation' and 'best_fitness' columns.")

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Best fitness over generations
    ax1.plot(df['generation'], df['best_fitness'], marker='o', linestyle='-', color='b', label='Best Fitness', markersize=3)
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Best Fitness')
    ax1.set_title('Generation vs Best Fitness')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()
    
    # Plot 2: Average fitness over generations
    if 'avg_fitness' in df.columns:
        ax2.plot(df['generation'], df['avg_fitness'], marker='s', linestyle='-', color='r', label='Average Fitness', markersize=3)
        ax2.plot(df['generation'], df['best_fitness'], marker='o', linestyle='-', color='b', label='Best Fitness', markersize=3, alpha=0.7)
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Fitness')
        ax2.set_title('Generation vs Fitness (Best vs Average)')
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend()
    
    plt.tight_layout()
    
    # Auto-generate save path if not provided
    if save_path is None:
        save_path = csv_path.replace('.csv', '_fitness_plot.png')
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    
    # Close the figure to free memory
    plt.close()
    
    return fig


def analyze_fitness_data(csv_path):
    """
    Analyze fitness data to identify patterns and issues.
    
    Args:
        csv_path (str): Path to the CSV file containing fitness data.
    """
    df = pd.read_csv(csv_path)
    
    print(f"=== Fitness Analysis for {csv_path} ===")
    print(f"Total generations: {len(df)}")
    print(f"Best fitness achieved: {df['best_fitness'].max():.2f}")
    print(f"Worst fitness: {df['best_fitness'].min():.2f}")
    print(f"Final best fitness: {df['best_fitness'].iloc[-1]:.2f}")
    
    # Check for failed evaluations (high fitness values)
    failed_threshold = 999.0
    failed_count = (df['best_fitness'] >= failed_threshold).sum()
    failed_percentage = (failed_count / len(df)) * 100
    print(f"Failed evaluations (≥{failed_threshold}): {failed_count} ({failed_percentage:.1f}%)")
    
    # Check for improvement
    initial_fitness = df['best_fitness'].iloc[0]
    final_fitness = df['best_fitness'].iloc[-1]
    improvement = final_fitness - initial_fitness
    print(f"Overall improvement: {improvement:+.2f}")
    
    # Check for convergence
    last_10_fitness = df['best_fitness'].tail(10)
    fitness_std = last_10_fitness.std()
    print(f"Fitness stability (last 10 generations std): {fitness_std:.2f}")
    
    if fitness_std < 1.0:
        print("✓ Fitness appears to have converged")
    else:
        print("✗ Fitness still fluctuating - may not have converged")
    
    # Identify best generation
    best_gen = df['best_fitness'].idxmax()
    print(f"Best generation: {best_gen} (fitness: {df['best_fitness'].iloc[best_gen]:.2f})")
    
    return df


# Example usage:
if __name__ == "__main__":

    #df = analyze_fitness_data('results/bayes/CdS_bayes_history.csv')
    #plot_generation_vs_best_fitness('results/bayes/CdS_bayes_history.csv')

    #df = analyze_fitness_data('results/cma1/CdS_cma1_history.csv')
    #plot_generation_vs_best_fitness('results/cma1/CdS_cma1_history.csv')

    #df = analyze_fitness_data('results/pso/CdS_pso_history.csv')
    #plot_generation_vs_best_fitness('results/pso/CdS_pso_history.csv')

    df = analyze_fitness_data('results/ga/CdS_ga_history.csv')
    plot_generation_vs_best_fitness('results/ga/CdS_ga_history.csv')
