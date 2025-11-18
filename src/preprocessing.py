"""
Data preprocessing module for network analysis.

This module provides functions for cleaning, analyzing, and splitting
network graphs for link prediction and other downstream tasks.
"""

import os
import pickle
import logging
import random
from typing import Dict, List, Tuple, Set
import networkx as nx
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def remove_self_loops(G: nx.Graph) -> nx.Graph:
    """
    Remove self-loops from a graph.
    
    Creates a copy of the graph and removes all edges where the source
    and target nodes are the same.
    
    Args:
        G: NetworkX graph (Graph, DiGraph, MultiGraph, or MultiDiGraph)
    
    Returns:
        New graph with self-loops removed
    
    Raises:
        TypeError: If G is not a NetworkX graph
        ValueError: If graph is empty
    """
    # Validation
    if not isinstance(G, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise TypeError(f"Expected NetworkX graph, got {type(G)}")
    
    if G.number_of_nodes() == 0:
        raise ValueError("Cannot process empty graph")
    
    logger.info(f"Removing self-loops from graph with {G.number_of_nodes()} nodes")
    
    # Create a copy to avoid modifying the original
    G_cleaned = G.copy()
    
    # Count self-loops before removal
    self_loops_before = len(list(nx.selfloop_edges(G_cleaned)))
    
    # Remove self-loops
    G_cleaned.remove_edges_from(nx.selfloop_edges(G_cleaned))
    
    self_loops_removed = self_loops_before
    
    if self_loops_removed > 0:
        logger.info(f"Removed {self_loops_removed} self-loop(s)")
    else:
        logger.info("No self-loops found")
    
    return G_cleaned


def get_largest_component(G: nx.Graph) -> nx.Graph:
    """
    Extract the largest connected component from a graph.
    
    For directed graphs, uses the largest weakly connected component.
    For undirected graphs, uses the largest connected component.
    
    Args:
        G: NetworkX graph (Graph, DiGraph, MultiGraph, or MultiDiGraph)
    
    Returns:
        Subgraph containing only the largest connected component
    
    Raises:
        TypeError: If G is not a NetworkX graph
        ValueError: If graph is empty or has no edges
    """
    # Validation
    if not isinstance(G, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise TypeError(f"Expected NetworkX graph, got {type(G)}")
    
    if G.number_of_nodes() == 0:
        raise ValueError("Cannot process empty graph")
    
    if G.number_of_edges() == 0:
        raise ValueError("Graph has no edges, cannot extract connected component")
    
    logger.info(f"Extracting largest component from graph with {G.number_of_nodes()} nodes")
    
    # Determine if graph is directed
    is_directed = G.is_directed()
    
    # Get connected components
    if is_directed:
        components = list(nx.weakly_connected_components(G))
        logger.info(f"Found {len(components)} weakly connected components")
    else:
        components = list(nx.connected_components(G))
        logger.info(f"Found {len(components)} connected components")
    
    if not components:
        raise ValueError("No connected components found")
    
    # Find largest component
    largest_component = max(components, key=len)
    largest_size = len(largest_component)
    
    logger.info(f"Largest component has {largest_size} nodes "
                f"({100 * largest_size / G.number_of_nodes():.2f}% of total)")
    
    # Create subgraph
    G_largest = G.subgraph(largest_component).copy()
    
    logger.info(f"Largest component: {G_largest.number_of_nodes()} nodes, "
                f"{G_largest.number_of_edges()} edges")
    
    return G_largest


def basic_statistics(G: nx.Graph) -> Dict[str, float]:
    """
    Compute basic statistics for a graph.
    
    Calculates fundamental network metrics including node count, edge count,
    density, average degree, and number of connected components.
    
    Args:
        G: NetworkX graph (Graph, DiGraph, MultiGraph, or MultiDiGraph)
    
    Returns:
        Dictionary containing:
            - 'num_nodes': Number of nodes
            - 'num_edges': Number of edges
            - 'density': Graph density (0 to 1)
            - 'avg_degree': Average node degree
            - 'num_components': Number of connected components
    
    Raises:
        TypeError: If G is not a NetworkX graph
        ValueError: If graph is empty
    """
    # Validation
    if not isinstance(G, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise TypeError(f"Expected NetworkX graph, got {type(G)}")
    
    if G.number_of_nodes() == 0:
        raise ValueError("Cannot compute statistics for empty graph")
    
    logger.info("Computing basic graph statistics")
    
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    
    # Calculate density
    if num_nodes <= 1:
        density = 0.0
    else:
        if G.is_directed():
            max_edges = num_nodes * (num_nodes - 1)
        else:
            max_edges = num_nodes * (num_nodes - 1) / 2
        density = num_edges / max_edges if max_edges > 0 else 0.0
    
    # Calculate average degree
    if num_nodes == 0:
        avg_degree = 0.0
    else:
        degrees = dict(G.degree())
        avg_degree = sum(degrees.values()) / num_nodes
    
    # Count connected components
    if G.is_directed():
        num_components = nx.number_weakly_connected_components(G)
    else:
        num_components = nx.number_connected_components(G)
    
    stats = {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'density': density,
        'avg_degree': avg_degree,
        'num_components': num_components
    }
    
    logger.info(f"Statistics: {num_nodes:,} nodes, {num_edges:,} edges, "
                f"density={density:.6f}, avg_degree={avg_degree:.2f}, "
                f"components={num_components}")
    
    return stats


def create_train_test_split(
    G: nx.Graph,
    test_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[nx.Graph, List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Split graph edges into training and test sets for link prediction.
    
    Creates a train/test split by:
    1. Randomly sampling edges for test set (positive examples)
    2. Creating same number of negative examples (non-edges)
    3. Returning training graph with test edges removed
    
    Args:
        G: NetworkX undirected graph
        test_ratio: Proportion of edges to use for testing (default: 0.2)
        seed: Random seed for reproducibility (default: 42)
    
    Returns:
        Tuple containing:
            - G_train: Training graph with test edges removed
            - positive_test_edges: List of (u, v) tuples for positive test edges
            - negative_test_edges: List of (u, v) tuples for negative test edges
    
    Raises:
        TypeError: If G is not a NetworkX graph
        ValueError: If graph is empty, has no edges, is directed, or test_ratio is invalid
    """
    # Validation
    if not isinstance(G, (nx.Graph, nx.MultiGraph)):
        raise TypeError(f"Expected undirected NetworkX graph, got {type(G)}")
    
    if G.is_directed():
        raise ValueError("Graph must be undirected for train/test split")
    
    if G.number_of_nodes() == 0:
        raise ValueError("Cannot split empty graph")
    
    if G.number_of_edges() == 0:
        raise ValueError("Cannot split graph with no edges")
    
    if not (0 < test_ratio < 1):
        raise ValueError(f"test_ratio must be between 0 and 1, got {test_ratio}")
    
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    
    logger.info(f"Creating train/test split with test_ratio={test_ratio}, seed={seed}")
    
    # Get all edges as a list
    all_edges = list(G.edges())
    num_edges = len(all_edges)
    num_test_edges = int(num_edges * test_ratio)
    
    logger.info(f"Total edges: {num_edges:,}, Test edges: {num_test_edges:,}, "
                f"Train edges: {num_edges - num_test_edges:,}")
    
    # Randomly sample test edges (positive examples)
    positive_test_edges = random.sample(all_edges, num_test_edges)
    positive_test_set = set(positive_test_edges)
    
    # Create training graph by removing test edges
    G_train = G.copy()
    G_train.remove_edges_from(positive_test_edges)
    
    logger.info(f"Training graph: {G_train.number_of_nodes()} nodes, "
                f"{G_train.number_of_edges()} edges")
    
    # Generate negative examples (non-edges)
    logger.info("Generating negative examples (non-edges)...")
    negative_test_edges = _generate_negative_edges(
        G, positive_test_set, num_test_edges, seed
    )
    
    logger.info(f"Generated {len(negative_test_edges)} negative test edges")
    
    return G_train, positive_test_edges, negative_test_edges


def _generate_negative_edges(
    G: nx.Graph,
    exclude_edges: Set[Tuple[int, int]],
    num_negative: int,
    seed: int
) -> List[Tuple[int, int]]:
    """
    Generate negative examples (non-edges) for link prediction.
    
    Randomly samples pairs of nodes that are not connected by an edge.
    Excludes existing edges and edges in the exclude set.
    
    Args:
        G: NetworkX graph
        exclude_edges: Set of edges to exclude from negative examples
        num_negative: Number of negative examples to generate
        seed: Random seed
    
    Returns:
        List of (u, v) tuples representing non-edges
    """
    random.seed(seed)
    np.random.seed(seed)
    
    nodes = list(G.nodes())
    num_nodes = len(nodes)
    existing_edges = set(G.edges())
    
    # Create set of all edges to exclude (existing + test positive edges)
    all_excluded = existing_edges | exclude_edges
    
    negative_edges = []
    max_attempts = num_negative * 100  # Prevent infinite loops
    attempts = 0
    
    while len(negative_edges) < num_negative and attempts < max_attempts:
        attempts += 1
        
        # Randomly sample two nodes
        u, v = random.sample(nodes, 2)
        
        # Ensure u < v for undirected graphs (canonical form)
        if u > v:
            u, v = v, u
        
        edge = (u, v)
        
        # Check if this is a valid negative edge
        if edge not in all_excluded and u != v:
            negative_edges.append(edge)
            all_excluded.add(edge)  # Avoid duplicates
    
    if len(negative_edges) < num_negative:
        logger.warning(f"Could only generate {len(negative_edges)} negative edges "
                      f"out of {num_negative} requested after {attempts} attempts")
    
    return negative_edges


def save_splits(
    G_train: nx.Graph,
    pos_test: List[Tuple[int, int]],
    neg_test: List[Tuple[int, int]],
    output_dir: str
) -> None:
    """
    Save train/test splits to pickle files.
    
    Saves the training graph, positive test edges, and negative test edges
    to separate pickle files in the specified output directory.
    
    Args:
        G_train: Training graph
        pos_test: List of positive test edges (u, v) tuples
        neg_test: List of negative test edges (u, v) tuples
        output_dir: Directory where files will be saved
    
    Raises:
        TypeError: If inputs are not of correct types
        IOError: If files cannot be written
        ValueError: If output directory cannot be created
    """
    # Validation
    if not isinstance(G_train, (nx.Graph, nx.MultiGraph)):
        raise TypeError(f"Expected NetworkX graph for G_train, got {type(G_train)}")
    
    if not isinstance(pos_test, list):
        raise TypeError(f"Expected list for pos_test, got {type(pos_test)}")
    
    if not isinstance(neg_test, list):
        raise TypeError(f"Expected list for neg_test, got {type(neg_test)}")
    
    if not isinstance(output_dir, str):
        raise TypeError(f"Expected string for output_dir, got {type(output_dir)}")
    
    # Create output directory if it doesn't exist
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
    except OSError as e:
        raise ValueError(f"Cannot create output directory {output_dir}: {e}")
    
    # Define file paths
    train_graph_path = os.path.join(output_dir, "train_graph.pkl")
    pos_test_path = os.path.join(output_dir, "positive_test_edges.pkl")
    neg_test_path = os.path.join(output_dir, "negative_test_edges.pkl")
    
    try:
        # Save training graph
        logger.info(f"Saving training graph to {train_graph_path}")
        with open(train_graph_path, 'wb') as f:
            pickle.dump(G_train, f)
        logger.info(f"Saved: {G_train.number_of_nodes()} nodes, "
                   f"{G_train.number_of_edges()} edges")
        
        # Save positive test edges
        logger.info(f"Saving positive test edges to {pos_test_path}")
        with open(pos_test_path, 'wb') as f:
            pickle.dump(pos_test, f)
        logger.info(f"Saved: {len(pos_test)} positive test edges")
        
        # Save negative test edges
        logger.info(f"Saving negative test edges to {neg_test_path}")
        with open(neg_test_path, 'wb') as f:
            pickle.dump(neg_test, f)
        logger.info(f"Saved: {len(neg_test)} negative test edges")
        
        logger.info("All splits saved successfully")
    
    except IOError as e:
        logger.error(f"Failed to save splits: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while saving splits: {e}")
        raise


def main():
    """
    Main function demonstrating usage of the preprocessing module.
    """
    print("=" * 60)
    print("Network Preprocessing Module - Demo")
    print("=" * 60)
    
    try:
        # Example: Create a sample graph for demonstration
        print("\n1. Creating sample graph...")
        G = nx.erdos_renyi_graph(n=1000, p=0.01, seed=42)
        print(f"   Created graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Add some self-loops for demonstration
        G.add_edge(0, 0)
        G.add_edge(1, 1)
        print(f"   Added self-loops: {G.number_of_selfloops()} self-loops")
        
        # 2. Remove self-loops
        print("\n2. Removing self-loops...")
        G_cleaned = remove_self_loops(G)
        print(f"   After cleaning: {G_cleaned.number_of_nodes()} nodes, "
              f"{G_cleaned.number_of_edges()} edges")
        
        # 3. Get largest component
        print("\n3. Extracting largest component...")
        G_largest = get_largest_component(G_cleaned)
        print(f"   Largest component: {G_largest.number_of_nodes()} nodes, "
              f"{G_largest.number_of_edges()} edges")
        
        # 4. Compute statistics
        print("\n4. Computing basic statistics...")
        stats = basic_statistics(G_largest)
        print(f"   Statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"     {key}: {value:.6f}")
            else:
                print(f"     {key}: {value:,}")
        
        # 5. Create train/test split
        print("\n5. Creating train/test split...")
        G_train, pos_test, neg_test = create_train_test_split(
            G_largest, test_ratio=0.2, seed=42
        )
        print(f"   Training graph: {G_train.number_of_nodes()} nodes, "
              f"{G_train.number_of_edges()} edges")
        print(f"   Positive test edges: {len(pos_test)}")
        print(f"   Negative test edges: {len(neg_test)}")
        
        # 6. Save splits
        print("\n6. Saving splits...")
        output_dir = "data/processed"
        save_splits(G_train, pos_test, neg_test, output_dir)
        print(f"   Splits saved to {output_dir}/")
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)
    
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"\n✗ Error: {e}")
        raise


if __name__ == "__main__":
    main()

