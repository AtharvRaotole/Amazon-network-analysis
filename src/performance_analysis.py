"""
Performance analysis module for network algorithms.

This module provides functions for timing, comparing, and analyzing
the performance of network analysis algorithms.
"""

import os
import time
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Callable, Any
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available 
              else 'seaborn-darkgrid' if 'seaborn-darkgrid' in plt.style.available 
              else 'ggplot')
sns.set_palette("husl")


def time_algorithm(
    func: Callable,
    *args,
    **kwargs
) -> Tuple[Any, Dict[str, float]]:
    """
    Time the execution of an algorithm.
    
    Uses both wall-clock time (time.time()) and CPU time (time.process_time())
    for comprehensive timing information.
    
    Args:
        func: Function to time
        *args: Positional arguments for function
        **kwargs: Keyword arguments for function
    
    Returns:
        Tuple of (result, timing_dict) where timing_dict contains:
            - wall_time: Wall-clock time (seconds)
            - cpu_time: CPU time (seconds)
            - user_time: User CPU time (if available)
    """
    logger.info(f"Timing algorithm: {func.__name__}")
    
    # Wall-clock time
    wall_start = time.time()
    
    # CPU time
    cpu_start = time.process_time()
    
    # Execute function
    try:
        result = func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Algorithm {func.__name__} failed: {e}")
        raise
    
    # Get end times
    cpu_end = time.process_time()
    wall_end = time.time()
    
    # Calculate times
    wall_time = wall_end - wall_start
    cpu_time = cpu_end - cpu_start
    
    timing_info = {
        'wall_time': wall_time,
        'cpu_time': cpu_time,
        'function_name': func.__name__
    }
    
    logger.info(f"Algorithm {func.__name__} completed: wall_time={wall_time:.4f}s, cpu_time={cpu_time:.4f}s")
    
    return result, timing_info


def compare_algorithm_runtimes(
    algorithms_dict: Dict[str, Callable],
    G: Any,
    *args,
    **kwargs
) -> pd.DataFrame:
    """
    Compare runtimes of multiple algorithms.
    
    Args:
        algorithms_dict: Dictionary mapping algorithm names to functions
        G: Graph to run algorithms on
        *args: Additional positional arguments for algorithms
        **kwargs: Additional keyword arguments for algorithms
    
    Returns:
        DataFrame with columns: algorithm, wall_time, cpu_time, speedup
    """
    logger.info(f"Comparing runtimes of {len(algorithms_dict)} algorithms...")
    
    results = []
    baseline_time = None
    
    for algo_name, algo_func in tqdm(algorithms_dict.items(), desc="Running algorithms"):
        try:
            # Time the algorithm
            result, timing = time_algorithm(algo_func, G, *args, **kwargs)
            
            # Set baseline (first algorithm)
            if baseline_time is None:
                baseline_time = timing['wall_time']
                speedup = 1.0
            else:
                speedup = baseline_time / timing['wall_time'] if timing['wall_time'] > 0 else 0.0
            
            results.append({
                'algorithm': algo_name,
                'wall_time': timing['wall_time'],
                'cpu_time': timing['cpu_time'],
                'speedup': speedup,
                'status': 'success'
            })
        
        except Exception as e:
            logger.error(f"Algorithm {algo_name} failed: {e}")
            results.append({
                'algorithm': algo_name,
                'wall_time': np.nan,
                'cpu_time': np.nan,
                'speedup': np.nan,
                'status': f'failed: {str(e)}'
            })
    
    df = pd.DataFrame(results)
    df = df.sort_values('wall_time', ascending=True)
    
    logger.info(f"Comparison complete: {len(df)} algorithms tested")
    
    return df


def plot_runtime_comparison(
    timing_results: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    use_log_scale: bool = True
) -> None:
    """
    Plot runtime comparison of algorithms.
    
    Args:
        timing_results: DataFrame with algorithm timing results
        save_path: Optional path to save figure
        figsize: Figure size tuple (default: (10, 6))
        use_log_scale: Whether to use log scale for y-axis (default: True)
    
    Raises:
        ValueError: If timing_results is empty
    """
    if timing_results.empty:
        raise ValueError("timing_results is empty")
    
    logger.info("Plotting runtime comparison...")
    
    # Filter out failed algorithms
    df_plot = timing_results[timing_results['status'] == 'success'].copy()
    
    if df_plot.empty:
        logger.warning("No successful algorithms to plot")
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort by wall_time
    df_plot = df_plot.sort_values('wall_time', ascending=True)
    
    # Create bar chart
    x_pos = np.arange(len(df_plot))
    bars = ax.bar(x_pos, df_plot['wall_time'].values, color=plt.cm.viridis(np.linspace(0, 1, len(df_plot))))
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, df_plot['wall_time'].values)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
               f'{val:.2f}s', ha='center', va='bottom', fontsize=9)
    
    # Formatting
    ax.set_xlabel('Algorithm', fontsize=12)
    ax.set_ylabel('Wall-Clock Time (seconds)', fontsize=12)
    ax.set_title('Algorithm Runtime Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df_plot['algorithm'].values, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    if use_log_scale:
        ax.set_yscale('log')
        ax.set_ylabel('Wall-Clock Time (seconds, log scale)', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Runtime comparison plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_scalability_results(
    benchmark_results: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Plot scalability results (time vs graph size).
    
    Args:
        benchmark_results: DataFrame with benchmark results
        save_path: Optional path to save figure
        figsize: Figure size tuple (default: (10, 6))
    
    Raises:
        ValueError: If benchmark_results is empty
    """
    if benchmark_results.empty:
        raise ValueError("benchmark_results is empty")
    
    logger.info("Plotting scalability results...")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Group by algorithm
    for algo in benchmark_results['algorithm'].unique():
        algo_data = benchmark_results[benchmark_results['algorithm'] == algo].sort_values('num_nodes')
        ax.plot(algo_data['num_nodes'], algo_data['time'], marker='o', label=algo, linewidth=2, markersize=6)
    
    # Formatting
    ax.set_xlabel('Number of Nodes', fontsize=12)
    ax.set_ylabel('Execution Time (seconds)', fontsize=12)
    ax.set_title('Algorithm Scalability Analysis', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.set_xscale('log')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Scalability plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def generate_performance_report(
    all_timings: Dict[str, pd.DataFrame],
    output_path: str
) -> None:
    """
    Generate comprehensive performance report.
    
    Args:
        all_timings: Dictionary mapping analysis type to timing DataFrame
        output_path: Path to save report (markdown format)
    
    Raises:
        ValueError: If all_timings is empty
    """
    if not all_timings:
        raise ValueError("all_timings is empty")
    
    logger.info(f"Generating performance report...")
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("# Performance Analysis Report\n\n")
        f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        # Summary statistics
        f.write("## Summary Statistics\n\n")
        total_algorithms = sum(len(df) for df in all_timings.values())
        total_time = sum(df['wall_time'].sum() for df in all_timings.values() if 'wall_time' in df.columns)
        f.write(f"- **Total Algorithms Tested**: {total_algorithms}\n")
        f.write(f"- **Total Execution Time**: {total_time:.2f} seconds ({total_time/60:.2f} minutes)\n\n")
        
        # Detailed results for each analysis type
        for analysis_type, df in all_timings.items():
            f.write(f"## {analysis_type.replace('_', ' ').title()}\n\n")
            
            if df.empty:
                f.write("No results available.\n\n")
                continue
            
            # Filter successful runs
            df_success = df[df['status'] == 'success'].copy() if 'status' in df.columns else df.copy()
            
            if not df_success.empty:
                # Summary table
                f.write("### Summary\n\n")
                f.write(f"- **Fastest Algorithm**: {df_success.loc[df_success['wall_time'].idxmin(), 'algorithm']}\n")
                f.write(f"- **Slowest Algorithm**: {df_success.loc[df_success['wall_time'].idxmax(), 'algorithm']}\n")
                f.write(f"- **Average Time**: {df_success['wall_time'].mean():.4f} seconds\n")
                f.write(f"- **Total Time**: {df_success['wall_time'].sum():.4f} seconds\n\n")
                
                # Detailed table
                f.write("### Detailed Results\n\n")
                f.write(df_success.to_markdown(index=False))
                f.write("\n\n")
            else:
                f.write("No successful runs.\n\n")
        
        # Recommendations
        f.write("---\n\n")
        f.write("## Recommendations\n\n")
        f.write("Based on the performance analysis:\n\n")
        
        for analysis_type, df in all_timings.items():
            df_success = df[df['status'] == 'success'].copy() if 'status' in df.columns else df.copy()
            if not df_success.empty and 'wall_time' in df_success.columns:
                fastest = df_success.loc[df_success['wall_time'].idxmin()]
                f.write(f"- **{analysis_type.replace('_', ' ').title()}**: ")
                f.write(f"Use {fastest['algorithm']} for fastest execution ({fastest['wall_time']:.4f}s)\n")
        
        f.write("\n")
    
    logger.info(f"Performance report saved to {output_path}")


def main():
    """
    Main function demonstrating usage of the performance analysis module.
    """
    print("=" * 60)
    print("Performance Analysis Module - Demo")
    print("=" * 60)
    
    try:
        import networkx as nx
        from centrality_analysis import compute_pagerank, compute_degree_centrality
        
        # Create a sample graph
        print("\n1. Creating sample graph...")
        G = nx.erdos_renyi_graph(n=500, p=0.02, seed=42)
        print(f"   Created graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Test timing function
        print("\n2. Testing timing function...")
        result, timing = time_algorithm(compute_pagerank, G)
        print(f"   Wall time: {timing['wall_time']:.4f} seconds")
        print(f"   CPU time: {timing['cpu_time']:.4f} seconds")
        
        # Test comparison
        print("\n3. Comparing algorithm runtimes...")
        algorithms = {
            'PageRank': compute_pagerank,
            'Degree Centrality': compute_degree_centrality
        }
        comparison_df = compare_algorithm_runtimes(algorithms, G)
        print(comparison_df.to_string(index=False))
        
        # Test plotting
        print("\n4. Plotting runtime comparison...")
        plot_runtime_comparison(comparison_df, save_path='results/figures/runtime_comparison.png')
        print("   ✅ Plot saved")
        
        # Test scalability plotting
        print("\n5. Testing scalability plotting...")
        benchmark_data = pd.DataFrame({
            'algorithm': ['PageRank', 'PageRank', 'Degree', 'Degree'],
            'num_nodes': [100, 500, 100, 500],
            'time': [0.1, 0.5, 0.05, 0.2]
        })
        plot_scalability_results(benchmark_data, save_path='results/figures/scalability.png')
        print("   ✅ Plot saved")
        
        # Generate report
        print("\n6. Generating performance report...")
        all_timings = {
            'centrality_algorithms': comparison_df
        }
        generate_performance_report(all_timings, 'results/tables/performance_report.md')
        print("   ✅ Report generated")
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)
    
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"\n✗ Error: {e}")
        raise


if __name__ == "__main__":
    main()

