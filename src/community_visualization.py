"""
Community visualization module for network graphs.

This module provides functions for visualizing community detection results
including size distributions, modularity comparisons, evaluation metrics,
and network visualizations.
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import Dict, List, Set, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set matplotlib style for professional plots
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    try:
        plt.style.use('seaborn-darkgrid')
    except OSError:
        plt.style.use('ggplot')

sns.set_palette("husl")


def plot_community_size_distribution(
    communities_dict: Dict[str, List[Set]],
    save_path: Optional[str] = None,
    figsize: tuple = (14, 8)
) -> None:
    """
    Create histogram of community sizes for all algorithms and ground truth.
    
    Args:
        communities_dict: Dictionary mapping algorithm name to list of community sets
                         Should include keys like 'louvain', 'label_propagation', 
                         'greedy_modularity', 'ground_truth'
        save_path: Optional path to save the figure (default: None)
        figsize: Figure size tuple (default: (14, 8))
    
    Raises:
        ValueError: If communities_dict is empty
        IOError: If figure cannot be saved
    """
    if not communities_dict:
        raise ValueError("communities_dict is empty")
    
    logger.info("Creating community size distribution plot...")
    
    # Prepare data
    plot_data = []
    for algo_name, communities in communities_dict.items():
        if communities is not None:
            sizes = [len(comm) for comm in communities]
            for size in sizes:
                plot_data.append({
                    'Algorithm': algo_name.replace('_', ' ').title(),
                    'Community Size': size
                })
    
    if not plot_data:
        raise ValueError("No valid community data to plot")
    
    df = pd.DataFrame(plot_data)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Community Size Distribution', fontsize=16, fontweight='bold', y=0.995)
    
    algorithms = df['Algorithm'].unique()
    colors = sns.color_palette("husl", len(algorithms))
    color_map = {algo: color for algo, color in zip(algorithms, colors)}
    
    # Plot 1: Histogram (linear scale)
    ax1 = axes[0, 0]
    for algo in algorithms:
        algo_data = df[df['Algorithm'] == algo]
        ax1.hist(algo_data['Community Size'], bins=50, alpha=0.6, 
                label=algo, color=color_map[algo], edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Community Size', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('Linear Scale', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Histogram (log scale)
    ax2 = axes[0, 1]
    for algo in algorithms:
        algo_data = df[df['Algorithm'] == algo]
        ax2.hist(algo_data['Community Size'], bins=50, alpha=0.6,
                label=algo, color=color_map[algo], edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Community Size (log scale)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency (log scale)', fontsize=12, fontweight='bold')
    ax2.set_title('Log Scale', fontsize=13, fontweight='bold')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, which='both')
    
    # Plot 3: Box plot
    ax3 = axes[1, 0]
    sns.boxplot(data=df, x='Algorithm', y='Community Size', ax=ax3, palette='husl')
    ax3.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Community Size', fontsize=12, fontweight='bold')
    ax3.set_title('Box Plot Comparison', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
    
    # Plot 4: Violin plot
    ax4 = axes[1, 1]
    sns.violinplot(data=df, x='Algorithm', y='Community Size', ax=ax4, palette='husl')
    ax4.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Community Size', fontsize=12, fontweight='bold')
    ax4.set_title('Violin Plot Comparison', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save figure if path provided
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")
        except IOError as e:
            logger.error(f"Failed to save figure: {e}")
            raise
    else:
        plt.show()
    
    plt.close()


def plot_modularity_comparison(
    results: Dict,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6)
) -> None:
    """
    Create bar chart comparing modularity scores across algorithms.
    
    Args:
        results: Dictionary from run_all_community_detection() or similar
                Should contain modularity scores for each algorithm
        save_path: Optional path to save the figure (default: None)
        figsize: Figure size tuple (default: (10, 6))
    
    Raises:
        ValueError: If results are invalid
        IOError: If figure cannot be saved
    """
    logger.info("Creating modularity comparison plot...")
    
    # Extract modularity scores
    modularity_data = []
    method_names = ['louvain', 'label_propagation', 'greedy_modularity']
    
    for method_name in method_names:
        if method_name in results and results[method_name] is not None:
            _, modularity = results[method_name]
            modularity_data.append({
                'Method': method_name.replace('_', ' ').title(),
                'Modularity': modularity
            })
    
    if not modularity_data:
        raise ValueError("No modularity data available in results")
    
    df = pd.DataFrame(modularity_data)
    df = df.sort_values('Modularity', ascending=False)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    bars = ax.bar(df['Method'], df['Modularity'], 
                  color=sns.color_palette("husl", len(df)), 
                  edgecolor='black', alpha=0.7, linewidth=1.5)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Modularity Score', fontsize=12, fontweight='bold')
    ax.set_title('Modularity Comparison Across Algorithms', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")
        except IOError as e:
            logger.error(f"Failed to save figure: {e}")
            raise
    else:
        plt.show()
    
    plt.close()


def plot_evaluation_metrics(
    evaluation_results: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6)
) -> None:
    """
    Create grouped bar chart comparing NMI and ARI scores across algorithms.
    
    Args:
        evaluation_results: DataFrame from compare_all_methods()
        save_path: Optional path to save the figure (default: None)
        figsize: Figure size tuple (default: (12, 6))
    
    Raises:
        TypeError: If evaluation_results is not a DataFrame
        ValueError: If DataFrame is empty or missing required columns
        IOError: If figure cannot be saved
    """
    if not isinstance(evaluation_results, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(evaluation_results)}")
    
    if evaluation_results.empty:
        raise ValueError("evaluation_results DataFrame is empty")
    
    required_columns = ['Method', 'NMI', 'ARI']
    missing = [col for col in required_columns if col not in evaluation_results.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    logger.info("Creating evaluation metrics comparison plot...")
    
    # Prepare data for grouped bar chart
    methods = evaluation_results['Method'].values
    nmi_values = evaluation_results['NMI'].values
    ari_values = evaluation_results['ARI'].values
    
    x = np.arange(len(methods))
    width = 0.35
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    bars1 = ax.bar(x - width/2, nmi_values, width, label='NMI', 
                   color='steelblue', edgecolor='black', alpha=0.7)
    bars2 = ax.bar(x + width/2, ari_values, width, label='ARI', 
                   color='coral', edgecolor='black', alpha=0.7)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Evaluation Metrics Comparison (NMI and ARI)', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")
        except IOError as e:
            logger.error(f"Failed to save figure: {e}")
            raise
    else:
        plt.show()
    
    plt.close()


def plot_largest_communities(
    G: nx.Graph,
    communities: List[Set],
    k: int = 5,
    save_path: Optional[str] = None,
    figsize: tuple = (16, 12),
    max_nodes_per_community: int = 100
) -> None:
    """
    Visualize k largest communities in the network.
    
    Uses spring layout and colors nodes by community. Samples nodes if
    communities are too large for visualization.
    
    Args:
        G: NetworkX graph
        communities: List of sets, each set contains node IDs in a community
        k: Number of largest communities to visualize (default: 5)
        save_path: Optional path to save the figure (default: None)
        figsize: Figure size tuple (default: (16, 12))
        max_nodes_per_community: Maximum nodes to show per community (default: 100)
    
    Raises:
        TypeError: If inputs are not of correct types
        ValueError: If inputs are invalid
        IOError: If figure cannot be saved
    """
    if not isinstance(G, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise TypeError(f"Expected NetworkX graph, got {type(G)}")
    
    if not isinstance(communities, list):
        raise TypeError(f"Expected list for communities, got {type(communities)}")
    
    if not communities:
        raise ValueError("Communities list is empty")
    
    logger.info(f"Visualizing {k} largest communities...")
    
    # Get k largest communities
    community_sizes = [(i, len(comm)) for i, comm in enumerate(communities)]
    community_sizes.sort(key=lambda x: x[1], reverse=True)
    largest_indices = [i for i, _ in community_sizes[:k]]
    
    # Create subgraph with nodes from largest communities
    all_nodes = set()
    for idx in largest_indices:
        comm = communities[idx]
        # Sample if too large
        if len(comm) > max_nodes_per_community:
            sampled = set(np.random.choice(list(comm), size=max_nodes_per_community, replace=False))
            all_nodes.update(sampled)
        else:
            all_nodes.update(comm)
    
    # Create subgraph
    G_sub = G.subgraph(all_nodes).copy()
    
    if G_sub.number_of_nodes() == 0:
        logger.warning("No nodes in subgraph for visualization")
        return
    
    # Create node-to-community mapping
    node_to_community = {}
    for idx in largest_indices:
        comm = communities[idx]
        for node in comm:
            if node in G_sub:
                node_to_community[node] = idx
    
    # Create figure with subplots
    n_cols = min(3, k)
    n_rows = (k + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle(f'Visualization of {k} Largest Communities', fontsize=16, fontweight='bold', y=0.995)
    
    if k == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if isinstance(axes, np.ndarray) else [axes]
    else:
        axes = axes.flatten()
    
    # Color palette
    colors = sns.color_palette("husl", k)
    
    for plot_idx, comm_idx in enumerate(largest_indices):
        ax = axes[plot_idx]
        
        # Get nodes in this community that are in subgraph
        comm_nodes = [n for n in communities[comm_idx] if n in G_sub]
        
        if not comm_nodes:
            ax.text(0.5, 0.5, 'No nodes in subgraph', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Community {comm_idx} (size: {len(communities[comm_idx])})', 
                        fontsize=11, fontweight='bold')
            ax.axis('off')
            continue
        
        # Create subgraph for this community
        G_comm = G_sub.subgraph(comm_nodes)
        
        # Layout
        try:
            pos = nx.spring_layout(G_comm, k=1, iterations=50, seed=42)
        except:
            pos = nx.random_layout(G_comm, seed=42)
        
        # Draw network
        nx.draw_networkx_nodes(G_comm, pos, ax=ax, node_color=colors[plot_idx],
                              node_size=50, alpha=0.7, edgecolors='black', linewidths=0.5)
        nx.draw_networkx_edges(G_comm, pos, ax=ax, alpha=0.3, width=0.5, edge_color='gray')
        
        ax.set_title(f'Community {comm_idx}\n(size: {len(communities[comm_idx])}, '
                    f'shown: {len(comm_nodes)})', fontsize=11, fontweight='bold')
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(k, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save figure if path provided
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")
        except IOError as e:
            logger.error(f"Failed to save figure: {e}")
            raise
    else:
        plt.show()
    
    plt.close()


def create_community_report(
    detection_results: Dict,
    evaluation_results: pd.DataFrame,
    output_path: str = "results/tables/community_report.html"
) -> None:
    """
    Generate comprehensive HTML report with all plots and metrics table.
    
    Args:
        detection_results: Dictionary from run_all_community_detection()
        evaluation_results: DataFrame from compare_all_methods()
        output_path: Path where HTML report will be saved
    
    Raises:
        ValueError: If output directory cannot be created
        IOError: If report cannot be saved
    """
    logger.info("Creating comprehensive community detection report...")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Generate all plots first
    plots_dir = os.path.join(os.path.dirname(output_path), 'community_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    try:
        # Plot 1: Modularity comparison
        plot_modularity_comparison(
            detection_results,
            save_path=os.path.join(plots_dir, 'modularity_comparison.png')
        )
        
        # Plot 2: Evaluation metrics
        if not evaluation_results.empty:
            plot_evaluation_metrics(
                evaluation_results,
                save_path=os.path.join(plots_dir, 'evaluation_metrics.png')
            )
        
        # Plot 3: Community size distribution (if ground truth available)
        communities_dict = {}
        for method in ['louvain', 'label_propagation', 'greedy_modularity']:
            if method in detection_results and detection_results[method] is not None:
                communities_dict[method] = detection_results[method][0]
        
        if communities_dict:
            plot_community_size_distribution(
                communities_dict,
                save_path=os.path.join(plots_dir, 'size_distribution.png')
            )
        
        # Plot 4: Largest communities visualization (for best method)
        best_method = None
        best_modularity = -1
        for method in ['louvain', 'label_propagation', 'greedy_modularity']:
            if method in detection_results and detection_results[method] is not None:
                _, mod = detection_results[method]
                if mod > best_modularity:
                    best_modularity = mod
                    best_method = method
        
        if best_method and detection_results[best_method] is not None:
            communities, _ = detection_results[best_method]
            # Need graph - would need to pass it, but for now skip if not available
            # plot_largest_communities(G, communities, k=5, 
            #                         save_path=os.path.join(plots_dir, 'largest_communities.png'))
        
        # Generate HTML content
        html_content = _generate_community_html_report(
            detection_results, evaluation_results, plots_dir
        )
        
        # Save HTML file
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to {output_path}")
        logger.info(f"All plots saved to {plots_dir}/")
    
    except Exception as e:
        logger.error(f"Failed to create report: {e}")
        raise


def _generate_community_html_report(
    detection_results: Dict,
    evaluation_results: pd.DataFrame,
    plots_dir: str
) -> str:
    """Generate HTML content for the community detection report."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Community Detection Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
            h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
            h2 { color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 5px; margin-top: 30px; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; background-color: white; }
            th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
            th { background-color: #3498db; color: white; font-weight: bold; }
            tr:nth-child(even) { background-color: #f2f2f2; }
            tr:hover { background-color: #e8f4f8; }
            .stats { background-color: white; padding: 15px; border-radius: 5px; margin: 10px 0; 
                    border-left: 4px solid #3498db; }
            img { max-width: 100%; height: auto; margin: 20px 0; border: 1px solid #ddd; 
                 border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .plot-section { background-color: white; padding: 20px; margin: 20px 0; 
                          border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        </style>
    </head>
    <body>
        <h1>Community Detection Analysis Report</h1>
    """
    
    # Add summary statistics
    html += "<h2>Summary Statistics</h2>"
    html += "<div class='stats'>"
    
    if 'summary' in detection_results:
        for algo_name, algo_summary in detection_results['summary'].items():
            if algo_summary is not None:
                html += f"<p><strong>{algo_name.replace('_', ' ').title()}:</strong> "
                html += f"{algo_summary['num_communities']} communities, "
                html += f"Modularity={algo_summary['modularity']:.6f}, "
                html += f"Avg size={algo_summary['avg_size']:.2f}</p>"
    
    html += "</div>"
    
    # Add evaluation metrics table
    if not evaluation_results.empty:
        html += "<h2>Evaluation Metrics Comparison</h2>"
        html += evaluation_results.to_html(index=False, classes='table', escape=False)
    
    # Add timing information
    if 'timing' in detection_results:
        html += "<h2>Computation Timing</h2>"
        html += "<div class='stats'>"
        for algo_name, time_taken in detection_results['timing'].items():
            if time_taken is not None:
                html += f"<p><strong>{algo_name.replace('_', ' ').title()}:</strong> {time_taken:.2f} seconds</p>"
        html += "</div>"
    
    # Add plots
    html += "<h2>Visualizations</h2>"
    
    plot_files = [
        ('modularity_comparison.png', 'Modularity Comparison'),
        ('evaluation_metrics.png', 'Evaluation Metrics (NMI and ARI)'),
        ('size_distribution.png', 'Community Size Distribution')
    ]
    
    for plot_file, plot_title in plot_files:
        plot_path = os.path.join(plots_dir, plot_file)
        if os.path.exists(plot_path):
            html += f"<div class='plot-section'>"
            html += f"<h3>{plot_title}</h3>"
            html += f"<img src='{plot_path}' alt='{plot_title}'>"
            html += "</div>"
    
    html += """
        <h2>Generated Files</h2>
        <p>All plot files are saved in the community_plots/ directory.</p>
    </body>
    </html>
    """
    
    return html


def main():
    """
    Main function demonstrating usage of the community visualization module.
    """
    print("=" * 60)
    print("Community Visualization Module - Demo")
    print("=" * 60)
    
    try:
        import networkx as nx
        from community_detection import run_all_community_detection
        
        # Create a sample graph
        print("\n1. Creating sample graph...")
        G = nx.erdos_renyi_graph(n=500, p=0.02, seed=42)
        print(f"   Created graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Run detection
        print("\n2. Running community detection...")
        detection_results = run_all_community_detection(G, seed=42)
        
        # Create visualizations
        print("\n3. Creating visualizations...")
        os.makedirs("results/figures", exist_ok=True)
        
        plot_modularity_comparison(detection_results, 
                                  save_path="results/figures/community_modularity.png")
        print("   ✅ Modularity comparison plot created")
        
        # Create communities dict for size distribution
        communities_dict = {}
        for method in ['louvain', 'label_propagation', 'greedy_modularity']:
            if method in detection_results and detection_results[method] is not None:
                communities_dict[method] = detection_results[method][0]
        
        plot_community_size_distribution(communities_dict,
                                        save_path="results/figures/community_sizes.png")
        print("   ✅ Size distribution plot created")
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)
    
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"\n✗ Error: {e}")
        raise


if __name__ == "__main__":
    main()

