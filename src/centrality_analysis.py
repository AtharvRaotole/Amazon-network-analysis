"""
Centrality analysis module for network graphs.

This module provides functions for computing various centrality measures
including PageRank, degree centrality, betweenness centrality, and HITS.
"""

import os
import time
import logging
import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, Tuple, Optional
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_pagerank(
    G: nx.Graph,
    alpha: float = 0.85,
    max_iter: int = 100
) -> Dict[int, float]:
    """
    Compute PageRank centrality for all nodes in the graph.
    
    PageRank measures the importance of nodes based on the structure
    of incoming links, with a damping factor to handle disconnected components.
    
    Args:
        G: NetworkX graph (Graph, DiGraph, MultiGraph, or MultiDiGraph)
        alpha: Damping parameter for PageRank (default: 0.85)
        max_iter: Maximum number of iterations (default: 100)
    
    Returns:
        Dictionary mapping node ID to PageRank score
    
    Raises:
        TypeError: If G is not a NetworkX graph
        ValueError: If graph is empty or parameters are invalid
    """
    # Validation
    if not isinstance(G, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise TypeError(f"Expected NetworkX graph, got {type(G)}")
    
    if G.number_of_nodes() == 0:
        raise ValueError("Cannot compute PageRank for empty graph")
    
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
    
    if max_iter < 1:
        raise ValueError(f"max_iter must be positive, got {max_iter}")
    
    logger.info(f"Computing PageRank (alpha={alpha}, max_iter={max_iter})...")
    start_time = time.time()
    
    try:
        # Compute PageRank
        pagerank_scores = nx.pagerank(G, alpha=alpha, max_iter=max_iter)
        
        elapsed_time = time.time() - start_time
        logger.info(f"PageRank computed in {elapsed_time:.2f} seconds")
        
        return pagerank_scores
    
    except Exception as e:
        logger.error(f"Failed to compute PageRank: {e}")
        raise


def compute_degree_centrality(G: nx.Graph) -> Dict[int, float]:
    """
    Compute degree centrality for all nodes in the graph.
    
    Degree centrality is the normalized degree (number of connections)
    divided by the maximum possible degree.
    
    Args:
        G: NetworkX graph (Graph, DiGraph, MultiGraph, or MultiDiGraph)
    
    Returns:
        Dictionary mapping node ID to degree centrality score
    
    Raises:
        TypeError: If G is not a NetworkX graph
        ValueError: If graph is empty
    """
    # Validation
    if not isinstance(G, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise TypeError(f"Expected NetworkX graph, got {type(G)}")
    
    if G.number_of_nodes() == 0:
        raise ValueError("Cannot compute degree centrality for empty graph")
    
    logger.info("Computing degree centrality...")
    start_time = time.time()
    
    try:
        # Compute degree centrality
        degree_centrality = nx.degree_centrality(G)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Degree centrality computed in {elapsed_time:.2f} seconds")
        
        return degree_centrality
    
    except Exception as e:
        logger.error(f"Failed to compute degree centrality: {e}")
        raise


def compute_betweenness_centrality(
    G: nx.Graph,
    k: Optional[int] = None
) -> Dict[int, float]:
    """
    Compute betweenness centrality for nodes in the graph.
    
    Betweenness centrality measures how often a node lies on the shortest
    path between pairs of other nodes. For large graphs, uses sampling.
    
    Args:
        G: NetworkX graph (Graph, DiGraph, MultiGraph, or MultiDiGraph)
        k: Number of nodes to sample for approximation (None for exact computation)
           Recommended: k=1000 for graphs with ~300K nodes
    
    Returns:
        Dictionary mapping node ID to betweenness centrality score
    
    Raises:
        TypeError: If G is not a NetworkX graph
        ValueError: If graph is empty or k is invalid
    """
    # Validation
    if not isinstance(G, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise TypeError(f"Expected NetworkX graph, got {type(G)}")
    
    if G.number_of_nodes() == 0:
        raise ValueError("Cannot compute betweenness centrality for empty graph")
    
    if k is not None and k < 1:
        raise ValueError(f"k must be positive, got {k}")
    
    num_nodes = G.number_of_nodes()
    
    # Determine if we should use sampling
    if k is None or k >= num_nodes:
        logger.info("Computing betweenness centrality (exact computation)...")
        use_sampling = False
    else:
        logger.info(f"Computing betweenness centrality (sampling k={k} nodes)...")
        use_sampling = True
    
    start_time = time.time()
    
    try:
        if use_sampling:
            # Sample k nodes for approximation
            sample_nodes = np.random.choice(
                list(G.nodes()),
                size=min(k, num_nodes),
                replace=False
            )
            
            # Compute betweenness centrality with sampling
            betweenness = nx.betweenness_centrality(
                G,
                k=k,
                endpoints=False
            )
        else:
            # Exact computation
            betweenness = nx.betweenness_centrality(G)
        
        elapsed_time = time.time() - start_time
        
        if use_sampling:
            logger.info(f"Betweenness centrality computed (sampled) in {elapsed_time:.2f} seconds")
        else:
            logger.info(f"Betweenness centrality computed (exact) in {elapsed_time:.2f} seconds")
        
        return betweenness
    
    except Exception as e:
        logger.error(f"Failed to compute betweenness centrality: {e}")
        raise


def compute_hits(
    G: nx.Graph,
    max_iter: int = 100
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Compute HITS (Hyperlink-Induced Topic Search) algorithm.
    
    HITS computes two scores for each node:
    - Hub score: measures how well a node points to good authorities
    - Authority score: measures how well a node is pointed to by good hubs
    
    Args:
        G: NetworkX graph (should be directed, will convert if undirected)
        max_iter: Maximum number of iterations (default: 100)
    
    Returns:
        Tuple of (hubs_dict, authorities_dict) where each maps node ID to score
    
    Raises:
        TypeError: If G is not a NetworkX graph
        ValueError: If graph is empty or max_iter is invalid
    """
    # Validation
    if not isinstance(G, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise TypeError(f"Expected NetworkX graph, got {type(G)}")
    
    if G.number_of_nodes() == 0:
        raise ValueError("Cannot compute HITS for empty graph")
    
    if max_iter < 1:
        raise ValueError(f"max_iter must be positive, got {max_iter}")
    
    logger.info(f"Computing HITS (max_iter={max_iter})...")
    start_time = time.time()
    
    try:
        # HITS requires directed graph
        if not G.is_directed():
            logger.info("Converting undirected graph to directed for HITS...")
            G_directed = G.to_directed()
        else:
            G_directed = G
        
        # Compute HITS
        hubs, authorities = nx.hits(G_directed, max_iter=max_iter)
        
        elapsed_time = time.time() - start_time
        logger.info(f"HITS computed in {elapsed_time:.2f} seconds")
        
        return hubs, authorities
    
    except Exception as e:
        logger.error(f"Failed to compute HITS: {e}")
        raise


def get_top_k_nodes(
    centrality_dict: Dict[int, float],
    k: int = 100
) -> pd.DataFrame:
    """
    Get top-k nodes sorted by centrality score.
    
    Args:
        centrality_dict: Dictionary mapping node ID to centrality score
        k: Number of top nodes to return (default: 100)
    
    Returns:
        Pandas DataFrame with columns: [node_id, score, rank]
        Sorted by score in descending order
    
    Raises:
        ValueError: If k is invalid or dictionary is empty
    """
    if not centrality_dict:
        raise ValueError("Centrality dictionary is empty")
    
    if k < 1:
        raise ValueError(f"k must be positive, got {k}")
    
    # Convert to DataFrame and sort
    df = pd.DataFrame([
        {'node_id': node, 'score': score}
        for node, score in centrality_dict.items()
    ])
    
    # Sort by score descending
    df = df.sort_values('score', ascending=False)
    
    # Add rank
    df['rank'] = range(1, len(df) + 1)
    
    # Select top k
    top_k = df.head(k).copy()
    
    # Reorder columns
    top_k = top_k[['node_id', 'score', 'rank']]
    
    return top_k


def compare_centrality_measures(
    G: nx.Graph,
    k: int = 100,
    betweenness_k: Optional[int] = None
) -> Dict:
    """
    Compute all centrality measures and compare them.
    
    Computes PageRank, degree centrality, betweenness centrality, and HITS,
    then calculates correlations between measures and returns top-k nodes.
    
    Args:
        G: NetworkX graph
        k: Number of top nodes to return for each measure (default: 100)
        betweenness_k: Number of nodes to sample for betweenness (default: None)
                      Recommended: 1000 for large graphs
    
    Returns:
        Dictionary containing:
            - 'pagerank': PageRank scores dict
            - 'degree': Degree centrality scores dict
            - 'betweenness': Betweenness centrality scores dict
            - 'hubs': HITS hub scores dict
            - 'authorities': HITS authority scores dict
            - 'top_k_pagerank': DataFrame of top-k PageRank nodes
            - 'top_k_degree': DataFrame of top-k degree centrality nodes
            - 'top_k_betweenness': DataFrame of top-k betweenness nodes
            - 'top_k_hubs': DataFrame of top-k hub nodes
            - 'top_k_authorities': DataFrame of top-k authority nodes
            - 'correlation_matrix': Correlation matrix between measures
            - 'timing': Dictionary with timing information for each measure
    
    Raises:
        TypeError: If G is not a NetworkX graph
        ValueError: If graph is empty
    """
    # Validation
    if not isinstance(G, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise TypeError(f"Expected NetworkX graph, got {type(G)}")
    
    if G.number_of_nodes() == 0:
        raise ValueError("Cannot compare centrality measures for empty graph")
    
    logger.info("Comparing all centrality measures...")
    overall_start = time.time()
    
    results = {}
    timing = {}
    
    # 1. Compute PageRank
    start = time.time()
    pagerank = compute_pagerank(G)
    timing['pagerank'] = time.time() - start
    results['pagerank'] = pagerank
    
    # 2. Compute degree centrality
    start = time.time()
    degree = compute_degree_centrality(G)
    timing['degree'] = time.time() - start
    results['degree'] = degree
    
    # 3. Compute betweenness centrality
    start = time.time()
    betweenness = compute_betweenness_centrality(G, k=betweenness_k)
    timing['betweenness'] = time.time() - start
    results['betweenness'] = betweenness
    
    # 4. Compute HITS
    start = time.time()
    hubs, authorities = compute_hits(G)
    timing['hits'] = time.time() - start
    results['hubs'] = hubs
    results['authorities'] = authorities
    
    # 5. Get top-k nodes for each measure
    logger.info(f"Extracting top-{k} nodes for each measure...")
    results['top_k_pagerank'] = get_top_k_nodes(pagerank, k=k)
    results['top_k_degree'] = get_top_k_nodes(degree, k=k)
    results['top_k_betweenness'] = get_top_k_nodes(betweenness, k=k)
    results['top_k_hubs'] = get_top_k_nodes(hubs, k=k)
    results['top_k_authorities'] = get_top_k_nodes(authorities, k=k)
    
    # 6. Compute correlation matrix
    logger.info("Computing correlation matrix...")
    start = time.time()
    
    # Get common nodes (nodes present in all measures)
    common_nodes = set(pagerank.keys())
    common_nodes = common_nodes.intersection(set(degree.keys()))
    common_nodes = common_nodes.intersection(set(betweenness.keys()))
    common_nodes = common_nodes.intersection(set(hubs.keys()))
    common_nodes = common_nodes.intersection(set(authorities.keys()))
    
    # Create DataFrame with all measures
    correlation_df = pd.DataFrame({
        'pagerank': [pagerank.get(node, np.nan) for node in common_nodes],
        'degree': [degree.get(node, np.nan) for node in common_nodes],
        'betweenness': [betweenness.get(node, np.nan) for node in common_nodes],
        'hubs': [hubs.get(node, np.nan) for node in common_nodes],
        'authorities': [authorities.get(node, np.nan) for node in common_nodes]
    }, index=list(common_nodes))
    
    # Compute correlation matrix
    correlation_matrix = correlation_df.corr()
    timing['correlation'] = time.time() - start
    
    results['correlation_matrix'] = correlation_matrix
    results['timing'] = timing
    
    overall_time = time.time() - overall_start
    logger.info(f"All centrality measures compared in {overall_time:.2f} seconds")
    
    return results


def save_centrality_results(
    results: Dict,
    output_dir: str = "results/tables"
) -> None:
    """
    Save centrality analysis results to CSV files.
    
    Saves:
    - Individual centrality scores (CSV)
    - Top-k nodes for each measure (CSV)
    - Correlation matrix (CSV)
    - Timing information (text file)
    
    Args:
        results: Dictionary from compare_centrality_measures()
        output_dir: Directory where files will be saved (default: "results/tables")
    
    Raises:
        ValueError: If output directory cannot be created
        IOError: If files cannot be written
    """
    if not isinstance(output_dir, str):
        raise TypeError(f"Expected string for output_dir, got {type(output_dir)}")
    
    # Create output directory
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving centrality results to {output_dir}")
    except OSError as e:
        raise ValueError(f"Cannot create output directory {output_dir}: {e}")
    
    try:
        # Save individual centrality scores
        centrality_measures = ['pagerank', 'degree', 'betweenness', 'hubs', 'authorities']
        
        for measure in centrality_measures:
            if measure in results:
                df = pd.DataFrame([
                    {'node_id': node, 'score': score}
                    for node, score in results[measure].items()
                ])
                df = df.sort_values('score', ascending=False)
                
                filepath = os.path.join(output_dir, f"{measure}_centrality.csv")
                df.to_csv(filepath, index=False)
                logger.info(f"Saved: {filepath} ({len(df)} nodes)")
        
        # Save top-k nodes
        top_k_measures = [
            'top_k_pagerank', 'top_k_degree', 'top_k_betweenness',
            'top_k_hubs', 'top_k_authorities'
        ]
        
        for measure in top_k_measures:
            if measure in results and isinstance(results[measure], pd.DataFrame):
                filepath = os.path.join(output_dir, f"{measure}.csv")
                results[measure].to_csv(filepath, index=False)
                logger.info(f"Saved: {filepath}")
        
        # Save correlation matrix
        if 'correlation_matrix' in results:
            filepath = os.path.join(output_dir, "centrality_correlation_matrix.csv")
            results['correlation_matrix'].to_csv(filepath)
            logger.info(f"Saved: {filepath}")
        
        # Save timing information
        if 'timing' in results:
            filepath = os.path.join(output_dir, "centrality_timing.txt")
            with open(filepath, 'w') as f:
                f.write("Centrality Analysis Timing Information\n")
                f.write("=" * 50 + "\n\n")
                for measure, time_taken in results['timing'].items():
                    f.write(f"{measure.capitalize()}: {time_taken:.2f} seconds\n")
                total_time = sum(results['timing'].values())
                f.write(f"\nTotal time: {total_time:.2f} seconds\n")
            logger.info(f"Saved: {filepath}")
        
        logger.info("All centrality results saved successfully")
    
    except IOError as e:
        logger.error(f"Failed to save centrality results: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while saving results: {e}")
        raise


def main():
    """
    Main function demonstrating usage of the centrality analysis module.
    """
    print("=" * 60)
    print("Centrality Analysis Module - Demo")
    print("=" * 60)
    
    try:
        # Load a sample graph for demonstration
        print("\n1. Creating sample graph...")
        G = nx.erdos_renyi_graph(n=1000, p=0.01, seed=42)
        print(f"   Created graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Compare all centrality measures
        print("\n2. Computing all centrality measures...")
        results = compare_centrality_measures(G, k=10, betweenness_k=100)
        
        print("\n3. Top-10 nodes by PageRank:")
        print(results['top_k_pagerank'].to_string(index=False))
        
        print("\n4. Correlation matrix:")
        print(results['correlation_matrix'].round(3).to_string())
        
        print("\n5. Timing information:")
        for measure, time_taken in results['timing'].items():
            print(f"   {measure}: {time_taken:.2f} seconds")
        
        # Save results
        print("\n6. Saving results...")
        save_centrality_results(results, output_dir="results/tables")
        print("   Results saved to results/tables/")
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)
    
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"\n✗ Error: {e}")
        raise


if __name__ == "__main__":
    main()

