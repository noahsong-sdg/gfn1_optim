"""
Unified Command Line Interface for TBLite parameter optimization.
Replaces redundant main functions across all optimizer modules.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from common import setup_logging, PROJECT_ROOT, CONFIG_DIR, RESULTS_DIR
from base_optimizer import BaseOptimizer
from optimizers.ga import GeneralParameterGA, GAConfig
from optimizers.gad import PyGADOptimizer, PyGADConfig
from optimizers.pso import GeneralParameterPSO, PSOConfig
from optimizers.bayes_h import GeneralParameterBayesian, BayesianConfig
from optimizers.cma1 import GeneralParameterCMA, CMAConfig
from optimizers.cma2 import GeneralParameterCMA2, CMA2Config

logger = setup_logging(module_name="cli")

def create_optimizer(algorithm: str, 
                    system_name: str,
                    base_param_file: str,
                    reference_data: Optional[str] = None,
                    **kwargs) -> BaseOptimizer:
    """Create optimizer instance based on algorithm name"""
    
    optimizer_map = {
        'ga': (GeneralParameterGA, GAConfig),
        'gad': (PyGADOptimizer, PyGADConfig),
        'pso': (GeneralParameterPSO, PSOConfig),
        'bayes': (GeneralParameterBayesian, BayesianConfig),
        'cma1': (GeneralParameterCMA, CMAConfig),
        'cma2': (GeneralParameterCMA2, CMA2Config)
    }
    
    if algorithm not in optimizer_map:
        raise ValueError(f"Unknown algorithm: {algorithm}. Available: {list(optimizer_map.keys())}")
    
    optimizer_class, config_class = optimizer_map[algorithm]
    
    # Create config with any overrides
    config_kwargs = {k: v for k, v in kwargs.items() if hasattr(config_class, k)}
    logger.debug(f"Creating {config_class.__name__} with kwargs: {config_kwargs}")
    config = config_class(**config_kwargs)
    
    # Load reference data if provided
    ref_data = None
    if reference_data and Path(reference_data).exists():
        import pandas as pd
        ref_data = pd.read_csv(reference_data)
    
    # Filter out config parameters from kwargs to avoid passing them to constructor
    config_params = {k: v for k, v in kwargs.items() if hasattr(config_class, k)}
    constructor_params = {k: v for k, v in kwargs.items() if k not in ['algorithm', 'system_name', 'base_param_file', 'reference_data'] and k not in config_params}
    
    logger.debug(f"Creating {optimizer_class.__name__} with constructor params: {constructor_params}")
    
    return optimizer_class(
        system_name=system_name,
        base_param_file=base_param_file,
        reference_data=ref_data,
        config=config,
        **constructor_params
    )

def run_optimization(algorithm: str,
                    system_name: str,
                    base_param_file: str,
                    reference_data: Optional[str] = None,
                    output_dir: Optional[str] = None,
                    **kwargs) -> dict:
    """Run optimization with the specified algorithm"""
    
    logger.info(f"Starting {algorithm.upper()} optimization for {system_name}")
    
    # Create optimizer
    optimizer = create_optimizer(
        algorithm=algorithm,
        system_name=system_name,
        base_param_file=base_param_file,
        reference_data=reference_data,
        **kwargs
    )
    
    # Run optimization
    best_params = optimizer.optimize()
    logger.info(f"Best fitness: {best_params.get('fitness', 'N/A')}")
    logger.info(f"Number of failed evaluations: {optimizer.failed_evaluations}")
    logger.info(f"Number of successful evaluations: {optimizer.success_evaluations}")

    # Save results
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save optimized parameters
        param_filename = output_path / f"{system_name}_{algorithm}.toml"
        optimizer.save_best_parameters(str(param_filename))
        
        # Save fitness history
        history_filename = output_path / f"{system_name}_{algorithm}_history.csv"
        optimizer.save_fitness_history(str(history_filename))
        
        logger.info(f"Results saved to {output_path}")
    
    return best_params

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="TBLite Parameter Optimization CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/cli.py ga H2 config/gfn1-base.toml --output results/ga
  python src/cli.py pso Si2 config/gfn1-base.toml --generations 100 --output results/pso
  python src/cli.py --reference-data train_structs/results.csv
  python src/cli.py bayes CdS config/gfn1-base.toml --n_calls 200 --output results/bayes
  python src/cli.py pso BulkMaterials config/gfn1-base.toml --max-iterations 50 --output results/bulk
  python src/cli.py ga CompareBulk config/gfn1-base.toml --generations 100 --output results/compare
        """
    )
    
    # Required arguments
    parser.add_argument('algorithm', 
                       choices=['ga', 'gad','pso', 'bayes', 'cma1', 'cma2'],
                       help='Optimization algorithm to use')
    parser.add_argument('system_name',
                       help='Name of the system to optimize (e.g., H2, Si2, CdS, GaN)')
    parser.add_argument('base_param_file',
                       help='Path to base parameter TOML file')
    
    # Optional arguments
    parser.add_argument('--reference-data', '-r',
                       help='Path to reference data CSV file')
    parser.add_argument('--output-dir', '-o',
                       help='Output directory for results')
    parser.add_argument('--train-fraction', type=float, default=0.8,
                       help='Fraction of data for training (default: 0.8)')
    parser.add_argument('--spin', type=int, default=0,
                       help='Spin multiplicity (default: 0)')
    
    # Algorithm-specific arguments
    parser.add_argument('--generations', type=int,
                       help='Number of generations (GA)')
    parser.add_argument('--max-iterations', type=int,
                       help='Maximum iterations (PSO)')
    parser.add_argument('--n-calls', type=int,
                       help='Number of function calls (Bayesian)')
    parser.add_argument('--population-size', type=int,
                       help='Population size (GA/PSO)')
    parser.add_argument('--swarm-size', type=int,
                       help='Swarm size (PSO)')
    
    # Logging
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    
    # Validate inputs
    if not Path(args.base_param_file).exists():
        logger.error(f"Base parameter file not found: {args.base_param_file}")
        sys.exit(1)
    
    if args.reference_data and not Path(args.reference_data).exists():
        logger.error(f"Reference data file not found: {args.reference_data}")
        sys.exit(1)
    
    # Convert args to kwargs for optimizer
    kwargs = {
        'train_fraction': args.train_fraction,
        'spin': args.spin
    }
    
    # Add algorithm-specific parameters
    if args.generations:
        kwargs['generations'] = args.generations
    if args.max_iterations:
        kwargs['max_iterations'] = args.max_iterations
    if args.n_calls:
        kwargs['n_calls'] = args.n_calls
    if args.population_size:
        kwargs['population_size'] = args.population_size
    if args.swarm_size:
        kwargs['swarm_size'] = args.swarm_size
    
    try:
        # Run optimization
        best_params = run_optimization(
            algorithm=args.algorithm,
            system_name=args.system_name,
            base_param_file=args.base_param_file,
            reference_data=args.reference_data,
            output_dir=args.output_dir,
            **kwargs
        )
        
        logger.info(f"Optimization completed successfully")
        logger.info(f"Best fitness: {best_params.get('fitness', 'N/A')}")
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
