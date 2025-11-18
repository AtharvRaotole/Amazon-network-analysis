"""
Exploratory data analysis module for network graphs.

This module provides functions for analyzing network properties,
visualizing degree distributions, and generating comprehensive
statistics reports.
"""

import os
import csv
import logging
from typing import Dict, Optional, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from tqdm import tqdm

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


def degree_distribution(G: nx.Graph) -> pd.Series:
    """
    Calculate degree distribution for all nodes in the graph.
    
    Computes the degree (number of connections) for each node and
    returns a pandas Series indexed by node ID.
    
    Args:
        G: NetworkX graph (Graph, DiGraph, MultiGraph, or MultiDiGraph)
    
    Returns:
        Pandas Series with node IDs as index and degrees as values
    
    Raises:
        TypeError: If G is not a NetworkX graph
        ValueError: If graph is empty
    """
    # Validation
    if not isinstance(G, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise TypeError(f"Expected NetworkX graph, got {type(G)}")
    
    if G.number_of_nodes() == 0:
        raise ValueError("Cannot compute degree distribution for empty graph")
    
    logger.info("Computing degree distribution")
    
    # Calculate degrees
    degrees = dict(G.degree())
    
    # Convert to pandas Series
    degree_series = pd.Series(degrees)
    degree_series.name = 'degree'
    
    logger.info(f"Computed degrees for {len(degree_series)} nodes")
    logger.info(f"Degree statistics: min={degree_series.min()}, "
                f"max={degree_series.max()}, mean={degree_series.mean():.2f}, "
                f"median={degree_series.median():.2f}")
    
    return degree_series


def plot_degree_distribution(
    G: nx.Graph,
    save_path: Optional[str] = None,
    figsize: tuple = (14, 6)
) -> None:
    """
    Plot degree distribution in both linear and log-log scales.
    
    Creates a figure with two subplots showing the degree distribution:
    - Left: Linear scale histogram
    - Right: Log-log scale scatter plot
    
    Args:
        G: NetworkX graph
        save_path: Optional path to save the figure (default: None)
        figsize: Figure size tuple (default: (14, 6))
    
    Raises:
        TypeError: If G is not a NetworkX graph
        ValueError: If graph is empty
        IOError: If figure cannot be saved
    """
    # Validation
    if not isinstance(G, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise TypeError(f"Expected NetworkX graph, got {type(G)}")
    
    if G.number_of_nodes() == 0:
        raise ValueError("Cannot plot degree distribution for empty graph")
    
    logger.info("Plotting degree distribution")
    
    # Get degree distribution
    degree_series = degree_distribution(G)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle('Degree Distribution', fontsize=16, fontweight='bold')
    
    # Left subplot: Linear scale histogram
    ax1.hist(degree_series.values, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.set_xlabel('Degree', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('Linear Scale', fontsize=13)
    ax1.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f'Mean: {degree_series.mean():.2f}\n'
    stats_text += f'Median: {degree_series.median():.2f}\n'
    stats_text += f'Max: {degree_series.max()}'
    ax1.text(0.7, 0.95, stats_text, transform=ax1.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Right subplot: Log-log scale
    # Count frequency of each degree value
    degree_counts = degree_series.value_counts().sort_index()
    
    # Filter out zero counts for log scale
    degree_counts = degree_counts[degree_counts > 0]
    
    ax2.scatter(degree_counts.index, degree_counts.values,
               alpha=0.6, s=50, color='coral', edgecolors='black', linewidth=0.5)
    ax2.set_xlabel('Degree (log scale)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency (log scale)', fontsize=12, fontweight='bold')
    ax2.set_title('Log-Log Scale', fontsize=13)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")
        except IOError as e:
            logger.error(f"Failed to save figure to {save_path}: {e}")
            raise
    else:
        plt.show()
    
    plt.close()


def compute_network_stats(
    G: nx.Graph,
    sample_size: Optional[int] = 10000
) -> Dict[str, float]:
    """
    Compute comprehensive network statistics.
    
    Calculates various network metrics including basic properties,
    clustering, triangles, and assortativity. For large graphs,
    uses sampling for computationally expensive metrics.
    
    Args:
        G: NetworkX graph (should be undirected for some metrics)
        sample_size: Number of nodes to sample for expensive computations
                    (default: 10000, None for no sampling)
    
    Returns:
        Dictionary containing:
            - 'num_nodes': Number of nodes
            - 'num_edges': Number of edges
            - 'density': Graph density
            - 'avg_clustering': Average clustering coefficient
            - 'num_triangles': Number of triangles
            - 'avg_degree': Average degree
            - 'degree_assortativity': Degree assortativity coefficient
    
    Raises:
        TypeError: If G is not a NetworkX graph
        ValueError: If graph is empty
    """
    # Validation
    if not isinstance(G, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise TypeError(f"Expected NetworkX graph, got {type(G)}")
    
    if G.number_of_nodes() == 0:
        raise ValueError("Cannot compute statistics for empty graph")
    
    logger.info("Computing network statistics")
    stats = {}
    
    # Basic statistics
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    stats['num_nodes'] = num_nodes
    stats['num_edges'] = num_edges
    
    # Density
    if num_nodes <= 1:
        density = 0.0
    else:
        if G.is_directed():
            max_edges = num_nodes * (num_nodes - 1)
        else:
            max_edges = num_nodes * (num_nodes - 1) / 2
        density = num_edges / max_edges if max_edges > 0 else 0.0
    stats['density'] = density
    
    # Average degree
    degrees = dict(G.degree())
    avg_degree = sum(degrees.values()) / num_nodes if num_nodes > 0 else 0.0
    stats['avg_degree'] = avg_degree
    
    # Clustering coefficient (sample if graph is large)
    logger.info("Computing average clustering coefficient...")
    if sample_size and num_nodes > sample_size:
        logger.info(f"Sampling {sample_size} nodes for clustering computation")
        sample_nodes = np.random.choice(list(G.nodes()), size=min(sample_size, num_nodes), replace=False)
        clustering_values = []
        for node in tqdm(sample_nodes, desc="Computing clustering"):
            try:
                clustering_values.append(nx.clustering(G, node))
            except:
                pass
        avg_clustering = np.mean(clustering_values) if clustering_values else 0.0
        stats['avg_clustering'] = avg_clustering
        stats['avg_clustering_sampled'] = True
    else:
        if not G.is_directed():
            avg_clustering = nx.average_clustering(G)
            stats['avg_clustering'] = avg_clustering
            stats['avg_clustering_sampled'] = False
        else:
            # For directed graphs, use undirected version
            G_undirected = G.to_undirected()
            avg_clustering = nx.average_clustering(G_undirected)
            stats['avg_clustering'] = avg_clustering
            stats['avg_clustering_sampled'] = False
    
    # Number of triangles (sample if graph is large)
    logger.info("Computing number of triangles...")
    if not G.is_directed():
        if sample_size and num_nodes > sample_size:
            logger.info(f"Estimating triangles using sampling")
            # Estimate triangles by sampling
            sample_nodes = np.random.choice(list(G.nodes()), size=min(sample_size, num_nodes), replace=False)
            triangles = 0
            for node in tqdm(sample_nodes, desc="Counting triangles"):
                neighbors = list(G.neighbors(node))
                for i, n1 in enumerate(neighbors):
                    for n2 in neighbors[i+1:]:
                        if G.has_edge(n1, n2):
                            triangles += 1
            # Scale up estimate
            num_triangles = int(triangles * (num_nodes / len(sample_nodes)))
            stats['num_triangles'] = num_triangles
            stats['num_triangles_sampled'] = True
        else:
            num_triangles = sum(nx.triangles(G).values()) // 3  # Each triangle counted 3 times
            stats['num_triangles'] = num_triangles
            stats['num_triangles_sampled'] = False
    else:
        # For directed graphs, convert to undirected
        G_undirected = G.to_undirected()
        if sample_size and num_nodes > sample_size:
            logger.info(f"Estimating triangles using sampling")
            sample_nodes = np.random.choice(list(G_undirected.nodes()), 
                                           size=min(sample_size, num_nodes), replace=False)
            triangles = 0
            for node in tqdm(sample_nodes, desc="Counting triangles"):
                neighbors = list(G_undirected.neighbors(node))
                for i, n1 in enumerate(neighbors):
                    for n2 in neighbors[i+1:]:
                        if G_undirected.has_edge(n1, n2):
                            triangles += 1
            num_triangles = int(triangles * (num_nodes / len(sample_nodes)))
            stats['num_triangles'] = num_triangles
            stats['num_triangles_sampled'] = True
        else:
            num_triangles = sum(nx.triangles(G_undirected).values()) // 3
            stats['num_triangles'] = num_triangles
            stats['num_triangles_sampled'] = False
    
    # Degree assortativity
    logger.info("Computing degree assortativity...")
    try:
        if not G.is_directed():
            assortativity = nx.degree_assortativity_coefficient(G)
        else:
            assortativity = nx.degree_assortativity_coefficient(G.to_undirected())
        stats['degree_assortativity'] = assortativity
    except Exception as e:
        logger.warning(f"Could not compute assortativity: {e}")
        stats['degree_assortativity'] = np.nan
    
    logger.info("Network statistics computed successfully")
    return stats


def generate_statistics_report(
    G: nx.Graph,
    communities: Optional[Dict[int, List[int]]] = None,
    output_path: str = "results/tables/network_statistics"
) -> None:
    """
    Generate comprehensive statistics report for the network.
    
    Computes all network statistics, includes community information
    if provided, and saves the report in both CSV and text formats.
    
    Args:
        G: NetworkX graph
        communities: Optional dictionary mapping community_id to list of nodes
        output_path: Base path for output files (without extension)
    
    Raises:
        TypeError: If inputs are not of correct types
        IOError: If files cannot be written
        ValueError: If output directory cannot be created
    """
    # Validation
    if not isinstance(G, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise TypeError(f"Expected NetworkX graph, got {type(G)}")
    
    if communities is not None and not isinstance(communities, dict):
        raise TypeError(f"Expected dict for communities, got {type(communities)}")
    
    logger.info("Generating statistics report")
    
    # Create output directory
    output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else '.'
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        raise ValueError(f"Cannot create output directory {output_dir}: {e}")
    
    # Compute network statistics
    stats = compute_network_stats(G)
    
    # Add community statistics if provided
    if communities:
        logger.info("Computing community statistics...")
        num_communities = len(communities)
        community_sizes = [len(nodes) for nodes in communities.values()]
        stats['num_communities'] = num_communities
        stats['avg_community_size'] = np.mean(community_sizes) if community_sizes else 0.0
        stats['max_community_size'] = max(community_sizes) if community_sizes else 0
        stats['min_community_size'] = min(community_sizes) if community_sizes else 0
    
    # Get degree distribution statistics
    degree_series = degree_distribution(G)
    stats['degree_min'] = degree_series.min()
    stats['degree_max'] = degree_series.max()
    stats['degree_mean'] = degree_series.mean()
    stats['degree_median'] = degree_series.median()
    stats['degree_std'] = degree_series.std()
    
    # Save as CSV
    csv_path = f"{output_path}.csv"
    try:
        logger.info(f"Saving CSV report to {csv_path}")
        df = pd.DataFrame([stats])
        df.to_csv(csv_path, index=False)
        logger.info("CSV report saved successfully")
    except IOError as e:
        logger.error(f"Failed to save CSV report: {e}")
        raise
    
    # Save as formatted text file
    txt_path = f"{output_path}.txt"
    try:
        logger.info(f"Saving text report to {txt_path}")
        with open(txt_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("NETWORK STATISTICS REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            # Basic properties
            f.write("BASIC PROPERTIES\n")
            f.write("-" * 70 + "\n")
            f.write(f"Number of nodes:           {stats['num_nodes']:,}\n")
            f.write(f"Number of edges:           {stats['num_edges']:,}\n")
            f.write(f"Density:                   {stats['density']:.8f}\n")
            f.write(f"Average degree:            {stats['avg_degree']:.4f}\n")
            f.write("\n")
            
            # Degree statistics
            f.write("DEGREE STATISTICS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Minimum degree:            {stats['degree_min']}\n")
            f.write(f"Maximum degree:            {stats['degree_max']}\n")
            f.write(f"Mean degree:               {stats['degree_mean']:.4f}\n")
            f.write(f"Median degree:             {stats['degree_median']:.4f}\n")
            f.write(f"Standard deviation:        {stats['degree_std']:.4f}\n")
            f.write("\n")
            
            # Clustering and triangles
            f.write("CLUSTERING & TRIANGLES\n")
            f.write("-" * 70 + "\n")
            f.write(f"Average clustering:        {stats['avg_clustering']:.6f}\n")
            if 'avg_clustering_sampled' in stats:
                f.write(f"  (Sampled: {stats['avg_clustering_sampled']})\n")
            f.write(f"Number of triangles:       {stats['num_triangles']:,}\n")
            if 'num_triangles_sampled' in stats:
                f.write(f"  (Sampled: {stats['num_triangles_sampled']})\n")
            f.write("\n")
            
            # Assortativity
            f.write("ASSORTATIVITY\n")
            f.write("-" * 70 + "\n")
            if not np.isnan(stats['degree_assortativity']):
                f.write(f"Degree assortativity:     {stats['degree_assortativity']:.6f}\n")
            else:
                f.write(f"Degree assortativity:      N/A\n")
            f.write("\n")
            
            # Community statistics (if available)
            if communities:
                f.write("COMMUNITY STATISTICS\n")
                f.write("-" * 70 + "\n")
                f.write(f"Number of communities:     {stats['num_communities']:,}\n")
                f.write(f"Average community size:    {stats['avg_community_size']:.2f}\n")
                f.write(f"Maximum community size:    {stats['max_community_size']}\n")
                f.write(f"Minimum community size:   {stats['min_community_size']}\n")
                f.write("\n")
            
            f.write("=" * 70 + "\n")
            f.write(f"Report generated successfully\n")
            f.write("=" * 70 + "\n")
        
        logger.info("Text report saved successfully")
    except IOError as e:
        logger.error(f"Failed to save text report: {e}")
        raise
    
    logger.info(f"Statistics report generated: {csv_path}, {txt_path}")


def main():
    """
    Main function demonstrating usage of the exploratory analysis module.
    """
    print("=" * 60)
    print("Exploratory Analysis Module - Demo")
    print("=" * 60)
    
    try:
        # Create a sample graph for demonstration
        print("\n1. Creating sample graph...")
        G = nx.erdos_renyi_graph(n=5000, p=0.001, seed=42)
        print(f"   Created graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # 2. Compute degree distribution
        print("\n2. Computing degree distribution...")
        degree_series = degree_distribution(G)
        print(f"   Degree statistics:")
        print(f"     Min: {degree_series.min()}, Max: {degree_series.max()}")
        print(f"     Mean: {degree_series.mean():.2f}, Median: {degree_series.median():.2f}")
        
        # 3. Plot degree distribution
        print("\n3. Plotting degree distribution...")
        plot_degree_distribution(G, save_path="results/figures/degree_distribution.png")
        print("   Plot saved to results/figures/degree_distribution.png")
        
        # 4. Compute network statistics
        print("\n4. Computing network statistics...")
        stats = compute_network_stats(G, sample_size=1000)
        print("   Statistics computed:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"     {key}: {value:.6f}")
            else:
                print(f"     {key}: {value:,}")
        
        # 5. Generate statistics report
        print("\n5. Generating statistics report...")
        # Create sample communities for demonstration
        communities = {
            0: list(range(0, 100)),
            1: list(range(100, 200)),
            2: list(range(200, 300))
        }
        generate_statistics_report(G, communities=communities)
        print("   Report saved to results/tables/network_statistics.csv and .txt")
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)
    
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"\n✗ Error: {e}")
        raise


if __name__ == "__main__":
    main()

