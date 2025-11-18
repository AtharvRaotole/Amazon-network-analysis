"""
Network properties computation module.

This module provides functions for computing various network properties
including basic statistics, degree distribution, clustering, path metrics,
and scale-free properties.
"""

import os
import time
import logging
import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, Tuple, Optional, List
from tqdm import tqdm
import random

# Try to import powerlaw library (optional)
try:
    import powerlaw
    POWERLAW_AVAILABLE = True
except ImportError:
    POWERLAW_AVAILABLE = False
    logging.warning("powerlaw library not available. Power-law fitting will be limited.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_basic_properties(G: nx.Graph) -> Dict:
    """
    Compute basic network properties.
    
    Args:
        G: NetworkX graph
    
    Returns:
        Dictionary with basic properties:
            - num_nodes: Number of nodes
            - num_edges: Number of edges
            - density: Graph density
            - avg_degree: Average degree
            - is_directed: Whether graph is directed
            - is_multigraph: Whether graph is multigraph
            - computation_time: Time taken to compute
    """
    start_time = time.time()
    
    logger.info("Computing basic network properties...")
    
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    
    # Compute density
    if num_nodes > 1:
        if G.is_directed():
            max_edges = num_nodes * (num_nodes - 1)
        else:
            max_edges = num_nodes * (num_nodes - 1) / 2
        density = num_edges / max_edges if max_edges > 0 else 0.0
    else:
        density = 0.0
    
    # Compute average degree
    if num_nodes > 0:
        if G.is_directed():
            avg_degree = (2 * num_edges) / num_nodes
        else:
            avg_degree = (2 * num_edges) / num_nodes
    else:
        avg_degree = 0.0
    
    computation_time = time.time() - start_time
    
    properties = {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'density': density,
        'avg_degree': avg_degree,
        'is_directed': G.is_directed(),
        'is_multigraph': G.is_multigraph(),
        'computation_time': computation_time
    }
    
    logger.info(f"Basic properties computed in {computation_time:.4f} seconds")
    
    return properties


def compute_degree_distribution(G: nx.Graph) -> Dict:
    """
    Compute degree distribution and statistics.
    
    Args:
        G: NetworkX graph
    
    Returns:
        Dictionary with:
            - degree_sequence: List of node degrees
            - degree_counts: Series with degree counts
            - mean_degree: Mean degree
            - median_degree: Median degree
            - std_degree: Standard deviation of degrees
            - min_degree: Minimum degree
            - max_degree: Maximum degree
            - power_law_alpha: Power-law exponent (if applicable)
            - power_law_fit: Whether power-law fit is good
            - computation_time: Time taken
    """
    start_time = time.time()
    
    logger.info("Computing degree distribution...")
    
    # Get degree sequence
    degrees = [d for n, d in G.degree()]
    degree_sequence = sorted(degrees, reverse=True)
    
    # Compute statistics
    degree_counts = pd.Series(degrees).value_counts().sort_index()
    
    mean_degree = np.mean(degrees)
    median_degree = np.median(degrees)
    std_degree = np.std(degrees)
    min_degree = min(degrees) if degrees else 0
    max_degree = max(degrees) if degrees else 0
    
    # Try to fit power law
    power_law_alpha = None
    power_law_fit = False
    
    if len(degree_sequence) > 10 and POWERLAW_AVAILABLE:
        try:
            # Fit power law
            fit = powerlaw.Fit(degree_sequence, discrete=True, verbose=False)
            power_law_alpha = fit.power_law.alpha
            # Test if power law is a good fit
            R, p = fit.distribution_compare('power_law', 'exponential', normalized_ratio=True)
            power_law_fit = p < 0.05  # Significant if p < 0.05
            logger.info(f"Power-law fit: alpha={power_law_alpha:.4f}, p-value={p:.4f}")
        except Exception as e:
            logger.warning(f"Power-law fitting failed: {e}")
    elif not POWERLAW_AVAILABLE:
        # Simple heuristic: check if distribution is heavy-tailed
        if len(degree_sequence) > 10:
            # Check if max degree >> mean degree (heuristic for power-law)
            if max_degree > 3 * mean_degree:
                power_law_fit = True
                logger.info("Heuristic suggests power-law distribution (powerlaw library not available)")
    
    computation_time = time.time() - start_time
    
    result = {
        'degree_sequence': degree_sequence,
        'degree_counts': degree_counts,
        'mean_degree': mean_degree,
        'median_degree': median_degree,
        'std_degree': std_degree,
        'min_degree': min_degree,
        'max_degree': max_degree,
        'power_law_alpha': power_law_alpha,
        'power_law_fit': power_law_fit,
        'computation_time': computation_time
    }
    
    logger.info(f"Degree distribution computed in {computation_time:.4f} seconds")
    
    return result


def compute_clustering_metrics(
    G: nx.Graph,
    sample_size: int = 1000
) -> Dict:
    """
    Compute clustering metrics.
    
    Args:
        G: NetworkX graph
        sample_size: Number of nodes to sample for average clustering (default: 1000)
    
    Returns:
        Dictionary with:
            - avg_clustering: Average clustering coefficient
            - global_clustering: Global clustering coefficient (transitivity)
            - transitivity: Transitivity (same as global clustering)
            - num_sampled_nodes: Number of nodes sampled
            - computation_time: Time taken
    """
    start_time = time.time()
    
    logger.info(f"Computing clustering metrics (sample_size={sample_size})...")
    
    num_nodes = G.number_of_nodes()
    
    # Global clustering (transitivity) - always compute full
    try:
        global_clustering = nx.transitivity(G)
        transitivity = global_clustering
    except:
        global_clustering = 0.0
        transitivity = 0.0
        logger.warning("Could not compute global clustering")
    
    # Average clustering - sample if graph is large
    if num_nodes > sample_size:
        logger.info(f"Sampling {sample_size} nodes for average clustering...")
        sampled_nodes = random.sample(list(G.nodes()), min(sample_size, num_nodes))
        try:
            avg_clustering = nx.average_clustering(G, nodes=sampled_nodes)
            num_sampled = len(sampled_nodes)
        except:
            avg_clustering = 0.0
            num_sampled = 0
            logger.warning("Could not compute average clustering")
    else:
        try:
            avg_clustering = nx.average_clustering(G)
            num_sampled = num_nodes
        except:
            avg_clustering = 0.0
            num_sampled = 0
            logger.warning("Could not compute average clustering")
    
    computation_time = time.time() - start_time
    
    result = {
        'avg_clustering': avg_clustering,
        'global_clustering': global_clustering,
        'transitivity': transitivity,
        'num_sampled_nodes': num_sampled,
        'computation_time': computation_time
    }
    
    logger.info(f"Clustering metrics computed in {computation_time:.4f} seconds")
    
    return result


def compute_path_metrics(
    G: nx.Graph,
    sample_size: int = 1000
) -> Dict:
    """
    Compute path-related metrics.
    
    Args:
        G: NetworkX graph
        sample_size: Number of node pairs to sample (default: 1000)
    
    Returns:
        Dictionary with:
            - avg_shortest_path: Average shortest path length
            - diameter: Graph diameter (approximate)
            - effective_diameter: Effective diameter (90th percentile)
            - num_sampled_pairs: Number of pairs sampled
            - is_connected: Whether graph is connected
            - computation_time: Time taken
    """
    start_time = time.time()
    
    logger.info(f"Computing path metrics (sample_size={sample_size})...")
    
    # Check if graph is connected
    if G.is_directed():
        is_connected = nx.is_strongly_connected(G)
    else:
        is_connected = nx.is_connected(G)
    
    if not is_connected:
        logger.warning("Graph is not connected. Computing metrics on largest component...")
        if G.is_directed():
            largest_cc = max(nx.strongly_connected_components(G), key=len)
        else:
            largest_cc = max(nx.connected_components(G), key=len)
        G_sub = G.subgraph(largest_cc).copy()
    else:
        G_sub = G
    
    num_nodes = G_sub.number_of_nodes()
    
    if num_nodes < 2:
        return {
            'avg_shortest_path': 0.0,
            'diameter': 0,
            'effective_diameter': 0.0,
            'num_sampled_pairs': 0,
            'is_connected': is_connected,
            'computation_time': time.time() - start_time
        }
    
    # Sample node pairs for path computation
    nodes_list = list(G_sub.nodes())
    
    if num_nodes * (num_nodes - 1) / 2 > sample_size:
        logger.info(f"Sampling {sample_size} node pairs for path computation...")
        pairs = []
        attempts = 0
        max_attempts = sample_size * 10
        
        while len(pairs) < sample_size and attempts < max_attempts:
            attempts += 1
            u, v = random.sample(nodes_list, 2)
            if u != v and (u, v) not in pairs and (v, u) not in pairs:
                pairs.append((u, v))
        
        num_sampled = len(pairs)
    else:
        # Use all pairs
        pairs = [(u, v) for u in nodes_list for v in nodes_list if u < v]
        num_sampled = len(pairs)
    
    # Compute shortest paths
    path_lengths = []
    for u, v in tqdm(pairs, desc="Computing shortest paths"):
        try:
            if G_sub.is_directed():
                path_length = nx.shortest_path_length(G_sub, u, v)
            else:
                path_length = nx.shortest_path_length(G_sub, u, v)
            path_lengths.append(path_length)
        except nx.NetworkXNoPath:
            continue
    
    if not path_lengths:
        avg_shortest_path = 0.0
        diameter = 0
        effective_diameter = 0.0
    else:
        avg_shortest_path = np.mean(path_lengths)
        diameter = max(path_lengths) if path_lengths else 0
        
        # Effective diameter (90th percentile)
        if path_lengths:
            sorted_lengths = sorted(path_lengths)
            percentile_90_idx = int(0.9 * len(sorted_lengths))
            effective_diameter = sorted_lengths[percentile_90_idx] if percentile_90_idx < len(sorted_lengths) else sorted_lengths[-1]
        else:
            effective_diameter = 0.0
    
    computation_time = time.time() - start_time
    
    result = {
        'avg_shortest_path': avg_shortest_path,
        'diameter': diameter,
        'effective_diameter': effective_diameter,
        'num_sampled_pairs': num_sampled,
        'is_connected': is_connected,
        'computation_time': computation_time
    }
    
    logger.info(f"Path metrics computed in {computation_time:.4f} seconds")
    
    return result


def compute_component_analysis(G: nx.Graph) -> Dict:
    """
    Compute connected component analysis.
    
    Args:
        G: NetworkX graph
    
    Returns:
        Dictionary with:
            - num_components: Number of connected components
            - largest_component_size: Size of largest component
            - component_sizes: List of component sizes
            - component_size_distribution: Series with size counts
            - computation_time: Time taken
    """
    start_time = time.time()
    
    logger.info("Computing component analysis...")
    
    if G.is_directed():
        components = list(nx.strongly_connected_components(G))
    else:
        components = list(nx.connected_components(G))
    
    num_components = len(components)
    component_sizes = [len(c) for c in components]
    
    if component_sizes:
        largest_component_size = max(component_sizes)
        component_size_distribution = pd.Series(component_sizes).value_counts().sort_index()
    else:
        largest_component_size = 0
        component_size_distribution = pd.Series()
    
    computation_time = time.time() - start_time
    
    result = {
        'num_components': num_components,
        'largest_component_size': largest_component_size,
        'component_sizes': component_sizes,
        'component_size_distribution': component_size_distribution,
        'computation_time': computation_time
    }
    
    logger.info(f"Component analysis computed in {computation_time:.4f} seconds")
    
    return result


def compute_assortativity(G: nx.Graph) -> Dict:
    """
    Compute degree assortativity coefficient.
    
    Args:
        G: NetworkX graph
    
    Returns:
        Dictionary with:
            - assortativity: Degree assortativity coefficient
            - computation_time: Time taken
    """
    start_time = time.time()
    
    logger.info("Computing degree assortativity...")
    
    try:
        assortativity = nx.degree_assortativity_coefficient(G)
    except:
        assortativity = 0.0
        logger.warning("Could not compute assortativity")
    
    computation_time = time.time() - start_time
    
    result = {
        'assortativity': assortativity,
        'computation_time': computation_time
    }
    
    logger.info(f"Assortativity computed in {computation_time:.4f} seconds")
    
    return result


def test_scale_free_property(G: nx.Graph) -> Dict:
    """
    Test if network follows scale-free (power-law) distribution.
    
    Args:
        G: NetworkX graph
    
    Returns:
        Dictionary with:
            - is_scale_free: Whether network is scale-free
            - power_law_alpha: Power-law exponent
            - p_value: P-value of power-law test
            - alternative_distribution: Best alternative distribution
            - computation_time: Time taken
    """
    start_time = time.time()
    
    logger.info("Testing scale-free property...")
    
    # Get degree sequence
    degrees = [d for n, d in G.degree()]
    degree_sequence = sorted([d for d in degrees if d > 0], reverse=True)
    
    is_scale_free = False
    power_law_alpha = None
    p_value = None
    alternative_distribution = None
    
    if len(degree_sequence) < 10:
        logger.warning("Not enough nodes for power-law test")
    elif POWERLAW_AVAILABLE:
        try:
            # Fit power law
            fit = powerlaw.Fit(degree_sequence, discrete=True, verbose=False)
            power_law_alpha = fit.power_law.alpha
            
            # Compare with exponential distribution
            R, p = fit.distribution_compare('power_law', 'exponential', normalized_ratio=True)
            p_value = p
            
            if R > 0 and p < 0.05:
                is_scale_free = True
                alternative_distribution = 'exponential'
            else:
                # Try comparing with log-normal
                R2, p2 = fit.distribution_compare('power_law', 'lognormal', normalized_ratio=True)
                if R2 > 0 and p2 < 0.05:
                    is_scale_free = True
                    alternative_distribution = 'lognormal'
                    p_value = p2
                else:
                    alternative_distribution = 'exponential' if abs(R) < abs(R2) else 'lognormal'
            
            logger.info(f"Power-law test: alpha={power_law_alpha:.4f}, p-value={p_value:.4f}, scale-free={is_scale_free}")
        except Exception as e:
            logger.warning(f"Power-law test failed: {e}")
    else:
        # Heuristic test
        if len(degree_sequence) > 10:
            mean_degree = np.mean(degree_sequence)
            max_degree = max(degree_sequence)
            # Heuristic: if max >> mean, likely power-law
            if max_degree > 5 * mean_degree:
                is_scale_free = True
                logger.info("Heuristic suggests scale-free property (powerlaw library not available)")
    
    computation_time = time.time() - start_time
    
    result = {
        'is_scale_free': is_scale_free,
        'power_law_alpha': power_law_alpha,
        'p_value': p_value,
        'alternative_distribution': alternative_distribution,
        'computation_time': computation_time
    }
    
    logger.info(f"Scale-free test completed in {computation_time:.4f} seconds")
    
    return result


def generate_network_report(
    G: nx.Graph,
    output_path: str,
    sample_size: int = 1000
) -> Dict:
    """
    Generate comprehensive network properties report.
    
    Args:
        G: NetworkX graph
        output_path: Path to save report (markdown format)
        sample_size: Sample size for expensive computations (default: 1000)
    
    Returns:
        Dictionary with all computed properties
    """
    logger.info(f"Generating comprehensive network report...")
    total_start_time = time.time()
    
    # Compute all properties
    logger.info("=" * 60)
    logger.info("Computing Network Properties")
    logger.info("=" * 60)
    
    basic_props = compute_basic_properties(G)
    degree_dist = compute_degree_distribution(G)
    clustering = compute_clustering_metrics(G, sample_size=sample_size)
    path_metrics = compute_path_metrics(G, sample_size=sample_size)
    components = compute_component_analysis(G)
    assortativity = compute_assortativity(G)
    scale_free = test_scale_free_property(G)
    
    total_time = time.time() - total_start_time
    
    # Combine all results
    all_properties = {
        'basic_properties': basic_props,
        'degree_distribution': degree_dist,
        'clustering': clustering,
        'path_metrics': path_metrics,
        'components': components,
        'assortativity': assortativity,
        'scale_free': scale_free,
        'total_computation_time': total_time
    }
    
    # Generate markdown report
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("# Network Properties Report\n\n")
        f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        # Basic Properties
        f.write("## 1. Basic Properties\n\n")
        f.write(f"- **Number of Nodes**: {basic_props['num_nodes']:,}\n")
        f.write(f"- **Number of Edges**: {basic_props['num_edges']:,}\n")
        f.write(f"- **Density**: {basic_props['density']:.6f}\n")
        f.write(f"- **Average Degree**: {basic_props['avg_degree']:.4f}\n")
        f.write(f"- **Is Directed**: {basic_props['is_directed']}\n")
        f.write(f"- **Is Multigraph**: {basic_props['is_multigraph']}\n")
        f.write(f"- **Computation Time**: {basic_props['computation_time']:.4f} seconds\n\n")
        
        # Degree Distribution
        f.write("## 2. Degree Distribution\n\n")
        f.write(f"- **Mean Degree**: {degree_dist['mean_degree']:.4f}\n")
        f.write(f"- **Median Degree**: {degree_dist['median_degree']:.4f}\n")
        f.write(f"- **Std Degree**: {degree_dist['std_degree']:.4f}\n")
        f.write(f"- **Min Degree**: {degree_dist['min_degree']}\n")
        f.write(f"- **Max Degree**: {degree_dist['max_degree']}\n")
        if degree_dist['power_law_alpha']:
            f.write(f"- **Power-Law Alpha**: {degree_dist['power_law_alpha']:.4f}\n")
            f.write(f"- **Power-Law Fit**: {degree_dist['power_law_fit']}\n")
        f.write(f"- **Computation Time**: {degree_dist['computation_time']:.4f} seconds\n\n")
        
        # Clustering
        f.write("## 3. Clustering Metrics\n\n")
        f.write(f"- **Average Clustering**: {clustering['avg_clustering']:.6f}\n")
        f.write(f"- **Global Clustering (Transitivity)**: {clustering['global_clustering']:.6f}\n")
        f.write(f"- **Nodes Sampled**: {clustering['num_sampled_nodes']:,}\n")
        f.write(f"- **Computation Time**: {clustering['computation_time']:.4f} seconds\n\n")
        
        # Path Metrics
        f.write("## 4. Path Metrics\n\n")
        f.write(f"- **Average Shortest Path**: {path_metrics['avg_shortest_path']:.4f}\n")
        f.write(f"- **Diameter**: {path_metrics['diameter']}\n")
        f.write(f"- **Effective Diameter (90th percentile)**: {path_metrics['effective_diameter']:.4f}\n")
        f.write(f"- **Is Connected**: {path_metrics['is_connected']}\n")
        f.write(f"- **Pairs Sampled**: {path_metrics['num_sampled_pairs']:,}\n")
        f.write(f"- **Computation Time**: {path_metrics['computation_time']:.4f} seconds\n\n")
        
        # Components
        f.write("## 5. Component Analysis\n\n")
        f.write(f"- **Number of Components**: {components['num_components']:,}\n")
        f.write(f"- **Largest Component Size**: {components['largest_component_size']:,}\n")
        if len(components['component_sizes']) > 0:
            f.write(f"- **Component Size Range**: {min(components['component_sizes'])} - {max(components['component_sizes'])}\n")
        f.write(f"- **Computation Time**: {components['computation_time']:.4f} seconds\n\n")
        
        # Assortativity
        f.write("## 6. Assortativity\n\n")
        f.write(f"- **Degree Assortativity**: {assortativity['assortativity']:.6f}\n")
        f.write(f"- **Computation Time**: {assortativity['computation_time']:.4f} seconds\n\n")
        
        # Scale-Free Property
        f.write("## 7. Scale-Free Property Test\n\n")
        f.write(f"- **Is Scale-Free**: {scale_free['is_scale_free']}\n")
        if scale_free['power_law_alpha']:
            f.write(f"- **Power-Law Alpha**: {scale_free['power_law_alpha']:.4f}\n")
        if scale_free['p_value']:
            f.write(f"- **P-Value**: {scale_free['p_value']:.4f}\n")
        if scale_free['alternative_distribution']:
            f.write(f"- **Alternative Distribution**: {scale_free['alternative_distribution']}\n")
        f.write(f"- **Computation Time**: {scale_free['computation_time']:.4f} seconds\n\n")
        
        # Summary
        f.write("---\n\n")
        f.write("## Summary\n\n")
        f.write(f"- **Total Computation Time**: {total_time:.4f} seconds ({total_time/60:.2f} minutes)\n")
        f.write(f"- **Sample Size Used**: {sample_size:,}\n")
    
    logger.info(f"Network report saved to {output_path}")
    logger.info(f"Total computation time: {total_time:.4f} seconds")
    
    return all_properties


def main():
    """
    Main function demonstrating usage of the network properties module.
    """
    print("=" * 60)
    print("Network Properties Module - Demo")
    print("=" * 60)
    
    try:
        import networkx as nx
        
        # Create a sample graph
        print("\n1. Creating sample graph...")
        G = nx.erdos_renyi_graph(n=1000, p=0.01, seed=42)
        print(f"   Created graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Compute basic properties
        print("\n2. Computing basic properties...")
        basic = compute_basic_properties(G)
        print(f"   Nodes: {basic['num_nodes']}, Edges: {basic['num_edges']}")
        print(f"   Density: {basic['density']:.6f}, Avg Degree: {basic['avg_degree']:.4f}")
        
        # Compute degree distribution
        print("\n3. Computing degree distribution...")
        degree = compute_degree_distribution(G)
        print(f"   Mean: {degree['mean_degree']:.4f}, Max: {degree['max_degree']}")
        
        # Compute clustering
        print("\n4. Computing clustering metrics...")
        clustering = compute_clustering_metrics(G, sample_size=500)
        print(f"   Avg Clustering: {clustering['avg_clustering']:.6f}")
        
        # Compute path metrics
        print("\n5. Computing path metrics...")
        paths = compute_path_metrics(G, sample_size=500)
        print(f"   Avg Shortest Path: {paths['avg_shortest_path']:.4f}")
        
        # Generate report
        print("\n6. Generating comprehensive report...")
        all_props = generate_network_report(G, 'results/tables/network_properties_report.md', sample_size=500)
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

