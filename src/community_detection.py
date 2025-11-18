"""
Community detection module for network graphs.

This module provides functions for detecting communities in networks using
various algorithms including Louvain, Label Propagation, and Greedy Modularity.
"""

import time
import logging
import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Set, Tuple, Optional
from tqdm import tqdm

# Import community detection algorithms
try:
    from networkx.algorithms import community
    HAS_COMMUNITY = True
except ImportError:
    try:
        import community as community_louvain
        HAS_COMMUNITY = False
        HAS_LOUVAIN = True
    except ImportError:
        HAS_COMMUNITY = False
        HAS_LOUVAIN = False
        logging.warning("Community detection algorithms not available. Install networkx>=3.0 or python-louvain")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def detect_communities_louvain(
    G: nx.Graph,
    resolution: float = 1.0,
    seed: Optional[int] = 42
) -> Tuple[List[Set], float]:
    """
    Detect communities using the Louvain algorithm.
    
    The Louvain algorithm is a greedy optimization method that maximizes modularity.
    It's fast and works well for large networks.
    
    Args:
        G: NetworkX undirected graph
        resolution: Resolution parameter (default: 1.0)
                   Higher values find smaller communities
        seed: Random seed for reproducibility (default: 42)
    
    Returns:
        Tuple of (communities, modularity) where:
            - communities: List of sets, each set contains node IDs in a community
            - modularity: Modularity score of the partition
    
    Raises:
        TypeError: If G is not a NetworkX graph
        ValueError: If graph is empty or directed
        ImportError: If community detection algorithms are not available
    """
    # Validation
    if not isinstance(G, (nx.Graph, nx.MultiGraph)):
        raise TypeError(f"Expected undirected NetworkX graph, got {type(G)}")
    
    if G.is_directed():
        raise ValueError("Louvain algorithm requires undirected graph")
    
    if G.number_of_nodes() == 0:
        raise ValueError("Cannot detect communities in empty graph")
    
    if not HAS_COMMUNITY and not HAS_LOUVAIN:
        raise ImportError("Community detection algorithms not available. Install networkx>=3.0 or python-louvain")
    
    logger.info(f"Detecting communities using Louvain algorithm (resolution={resolution})...")
    start_time = time.time()
    
    try:
        if HAS_COMMUNITY:
            # Use NetworkX community detection
            if seed is not None:
                np.random.seed(seed)
            communities = list(community.louvain_communities(G, resolution=resolution, seed=seed))
        else:
            # Use python-louvain package
            if seed is not None:
                np.random.seed(seed)
            partition = community_louvain.best_partition(G, resolution=resolution, random_state=seed)
            # Convert partition dict to list of sets
            communities_dict = {}
            for node, comm_id in partition.items():
                if comm_id not in communities_dict:
                    communities_dict[comm_id] = set()
                communities_dict[comm_id].add(node)
            communities = list(communities_dict.values())
        
        # Calculate modularity
        modularity = calculate_modularity(G, communities)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Louvain communities detected: {len(communities)} communities, "
                   f"modularity={modularity:.6f}, time={elapsed_time:.2f}s")
        
        return communities, modularity
    
    except Exception as e:
        logger.error(f"Failed to detect communities with Louvain: {e}")
        raise


def detect_communities_label_propagation(
    G: nx.Graph,
    seed: Optional[int] = 42
) -> Tuple[List[Set], float]:
    """
    Detect communities using the Label Propagation algorithm.
    
    Label Propagation is a fast algorithm that works by propagating labels
    through the network. It's efficient for large graphs.
    
    Args:
        G: NetworkX undirected graph
        seed: Random seed for reproducibility (default: 42)
    
    Returns:
        Tuple of (communities, modularity) where:
            - communities: List of sets, each set contains node IDs in a community
            - modularity: Modularity score of the partition
    
    Raises:
        TypeError: If G is not a NetworkX graph
        ValueError: If graph is empty or directed
        ImportError: If community detection algorithms are not available
    """
    # Validation
    if not isinstance(G, (nx.Graph, nx.MultiGraph)):
        raise TypeError(f"Expected undirected NetworkX graph, got {type(G)}")
    
    if G.is_directed():
        raise ValueError("Label Propagation algorithm requires undirected graph")
    
    if G.number_of_nodes() == 0:
        raise ValueError("Cannot detect communities in empty graph")
    
    if not HAS_COMMUNITY:
        raise ImportError("Label Propagation requires networkx>=3.0")
    
    logger.info("Detecting communities using Label Propagation algorithm...")
    start_time = time.time()
    
    try:
        if seed is not None:
            np.random.seed(seed)
        
        # Use NetworkX label propagation
        communities = list(community.label_propagation_communities(G))
        
        # Calculate modularity
        modularity = calculate_modularity(G, communities)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Label Propagation communities detected: {len(communities)} communities, "
                   f"modularity={modularity:.6f}, time={elapsed_time:.2f}s")
        
        return communities, modularity
    
    except Exception as e:
        logger.error(f"Failed to detect communities with Label Propagation: {e}")
        raise


def detect_communities_greedy_modularity(
    G: nx.Graph
) -> Tuple[List[Set], float]:
    """
    Detect communities using the Greedy Modularity algorithm.
    
    This algorithm greedily merges communities to maximize modularity.
    It's slower than Louvain but provides good results.
    
    Args:
        G: NetworkX undirected graph
    
    Returns:
        Tuple of (communities, modularity) where:
            - communities: List of sets, each set contains node IDs in a community
            - modularity: Modularity score of the partition
    
    Raises:
        TypeError: If G is not a NetworkX graph
        ValueError: If graph is empty or directed
        ImportError: If community detection algorithms are not available
    """
    # Validation
    if not isinstance(G, (nx.Graph, nx.MultiGraph)):
        raise TypeError(f"Expected undirected NetworkX graph, got {type(G)}")
    
    if G.is_directed():
        raise ValueError("Greedy Modularity algorithm requires undirected graph")
    
    if G.number_of_nodes() == 0:
        raise ValueError("Cannot detect communities in empty graph")
    
    if not HAS_COMMUNITY:
        raise ImportError("Greedy Modularity requires networkx>=3.0")
    
    logger.info("Detecting communities using Greedy Modularity algorithm...")
    start_time = time.time()
    
    try:
        # Use NetworkX greedy modularity
        communities = list(community.greedy_modularity_communities(G))
        
        # Calculate modularity
        modularity = calculate_modularity(G, communities)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Greedy Modularity communities detected: {len(communities)} communities, "
                   f"modularity={modularity:.6f}, time={elapsed_time:.2f}s")
        
        return communities, modularity
    
    except Exception as e:
        logger.error(f"Failed to detect communities with Greedy Modularity: {e}")
        raise


def communities_to_dict(communities: List[Set]) -> Dict[int, int]:
    """
    Convert list of community sets to node-to-community mapping.
    
    Args:
        communities: List of sets, each set contains node IDs in a community
    
    Returns:
        Dictionary mapping node ID to community ID (0-indexed)
    
    Raises:
        TypeError: If communities is not a list
        ValueError: If communities is empty
    """
    if not isinstance(communities, list):
        raise TypeError(f"Expected list, got {type(communities)}")
    
    if not communities:
        raise ValueError("Communities list is empty")
    
    node_to_community = {}
    for community_id, community_set in enumerate(communities):
        for node in community_set:
            if node in node_to_community:
                logger.warning(f"Node {node} appears in multiple communities")
            node_to_community[node] = community_id
    
    return node_to_community


def calculate_modularity(G: nx.Graph, communities: List[Set]) -> float:
    """
    Calculate modularity score for a given community partition.
    
    Modularity measures the quality of a community partition. Higher values
    indicate better community structure.
    
    Args:
        G: NetworkX graph
        communities: List of sets, each set contains node IDs in a community
    
    Returns:
        Modularity score (typically between -0.5 and 1.0)
    
    Raises:
        TypeError: If inputs are not of correct types
        ValueError: If graph is empty or communities is empty
        ImportError: If community detection algorithms are not available
    """
    if not isinstance(G, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise TypeError(f"Expected NetworkX graph, got {type(G)}")
    
    if G.number_of_nodes() == 0:
        raise ValueError("Cannot calculate modularity for empty graph")
    
    if not communities:
        raise ValueError("Communities list is empty")
    
    if not HAS_COMMUNITY:
        raise ImportError("Modularity calculation requires networkx>=3.0")
    
    try:
        # Convert communities to partition format if needed
        # NetworkX modularity expects a list of sets
        modularity = community.modularity(G, communities)
        return modularity
    
    except Exception as e:
        logger.error(f"Failed to calculate modularity: {e}")
        raise


def get_community_sizes(communities: List[Set]) -> pd.Series:
    """
    Get sorted list of community sizes.
    
    Args:
        communities: List of sets, each set contains node IDs in a community
    
    Returns:
        Pandas Series with community sizes, sorted in descending order
    
    Raises:
        TypeError: If communities is not a list
        ValueError: If communities is empty
    """
    if not isinstance(communities, list):
        raise TypeError(f"Expected list, got {type(communities)}")
    
    if not communities:
        raise ValueError("Communities list is empty")
    
    sizes = [len(comm) for comm in communities]
    sizes_series = pd.Series(sizes, name='community_size')
    sizes_series = sizes_series.sort_values(ascending=False)
    
    return sizes_series


def run_all_community_detection(
    G: nx.Graph,
    louvain_resolution: float = 1.0,
    seed: Optional[int] = 42
) -> Dict:
    """
    Run all community detection algorithms and compare results.
    
    Executes Louvain, Label Propagation, and Greedy Modularity algorithms,
    times each one, and returns comprehensive results.
    
    Args:
        G: NetworkX undirected graph
        louvain_resolution: Resolution parameter for Louvain (default: 1.0)
        seed: Random seed for reproducibility (default: 42)
    
    Returns:
        Dictionary containing:
            - 'louvain': Tuple of (communities, modularity)
            - 'label_propagation': Tuple of (communities, modularity)
            - 'greedy_modularity': Tuple of (communities, modularity)
            - 'timing': Dictionary with timing for each algorithm
            - 'summary': Dictionary with summary statistics
    
    Raises:
        TypeError: If G is not a NetworkX graph
        ValueError: If graph is empty or directed
    """
    # Validation
    if not isinstance(G, (nx.Graph, nx.MultiGraph)):
        raise TypeError(f"Expected undirected NetworkX graph, got {type(G)}")
    
    if G.is_directed():
        raise ValueError("Community detection algorithms require undirected graph")
    
    if G.number_of_nodes() == 0:
        raise ValueError("Cannot detect communities in empty graph")
    
    logger.info("Running all community detection algorithms...")
    overall_start = time.time()
    
    results = {}
    timing = {}
    
    # 1. Louvain
    try:
        start = time.time()
        louvain_communities, louvain_modularity = detect_communities_louvain(
            G, resolution=louvain_resolution, seed=seed
        )
        timing['louvain'] = time.time() - start
        results['louvain'] = (louvain_communities, louvain_modularity)
        logger.info(f"✅ Louvain completed: {len(louvain_communities)} communities, "
                   f"modularity={louvain_modularity:.6f}")
    except Exception as e:
        logger.error(f"❌ Louvain failed: {e}")
        results['louvain'] = None
        timing['louvain'] = None
    
    # 2. Label Propagation
    try:
        start = time.time()
        lp_communities, lp_modularity = detect_communities_label_propagation(G, seed=seed)
        timing['label_propagation'] = time.time() - start
        results['label_propagation'] = (lp_communities, lp_modularity)
        logger.info(f"✅ Label Propagation completed: {len(lp_communities)} communities, "
                   f"modularity={lp_modularity:.6f}")
    except Exception as e:
        logger.error(f"❌ Label Propagation failed: {e}")
        results['label_propagation'] = None
        timing['label_propagation'] = None
    
    # 3. Greedy Modularity
    try:
        start = time.time()
        gm_communities, gm_modularity = detect_communities_greedy_modularity(G)
        timing['greedy_modularity'] = time.time() - start
        results['greedy_modularity'] = (gm_communities, gm_modularity)
        logger.info(f"✅ Greedy Modularity completed: {len(gm_communities)} communities, "
                   f"modularity={gm_modularity:.6f}")
    except Exception as e:
        logger.error(f"❌ Greedy Modularity failed: {e}")
        results['greedy_modularity'] = None
        timing['greedy_modularity'] = None
    
    # Create summary
    summary = {}
    for algo_name, algo_result in results.items():
        if algo_result is not None:
            communities, modularity = algo_result
            sizes = get_community_sizes(communities)
            summary[algo_name] = {
                'num_communities': len(communities),
                'modularity': modularity,
                'avg_size': sizes.mean(),
                'max_size': sizes.max(),
                'min_size': sizes.min(),
                'median_size': sizes.median()
            }
        else:
            summary[algo_name] = None
    
    results['timing'] = timing
    results['summary'] = summary
    
    overall_time = time.time() - overall_start
    logger.info(f"All community detection algorithms completed in {overall_time:.2f} seconds")
    
    return results


def main():
    """
    Main function demonstrating usage of the community detection module.
    """
    print("=" * 60)
    print("Community Detection Module - Demo")
    print("=" * 60)
    
    try:
        # Create a sample graph for demonstration
        print("\n1. Creating sample graph...")
        G = nx.erdos_renyi_graph(n=1000, p=0.01, seed=42)
        print(f"   Created graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Run all algorithms
        print("\n2. Running all community detection algorithms...")
        results = run_all_community_detection(G, seed=42)
        
        # Display results
        print("\n3. Results Summary:")
        print("=" * 60)
        for algo_name, algo_summary in results['summary'].items():
            if algo_summary is not None:
                print(f"\n{algo_name.upper()}:")
                print(f"  Communities: {algo_summary['num_communities']}")
                print(f"  Modularity: {algo_summary['modularity']:.6f}")
                print(f"  Avg size: {algo_summary['avg_size']:.2f}")
                print(f"  Size range: {algo_summary['min_size']} - {algo_summary['max_size']}")
        
        print("\n4. Timing Information:")
        print("=" * 60)
        for algo_name, time_taken in results['timing'].items():
            if time_taken is not None:
                print(f"  {algo_name}: {time_taken:.2f} seconds")
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)
    
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"\n✗ Error: {e}")
        raise


if __name__ == "__main__":
    main()

