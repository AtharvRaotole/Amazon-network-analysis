"""
Data loader module for Amazon Product Co-Purchasing Network dataset.

This module provides functions to download, load, and save the Amazon
co-purchasing network dataset from SNAP (Stanford Network Analysis Project).
"""

import os
import gzip
import pickle
import logging
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional
import networkx as nx
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# SNAP dataset URLs
GRAPH_URL = "https://snap.stanford.edu/data/bigdata/communities/com-amazon.ungraph.txt.gz"
COMMUNITIES_URL = "https://snap.stanford.edu/data/bigdata/communities/com-amazon.all.cmty.txt.gz"


def download_file(url: str, filepath: str, chunk_size: int = 8192) -> None:
    """
    Download a file from a URL with progress bar.
    
    Args:
        url: URL of the file to download
        filepath: Local path where the file will be saved
        chunk_size: Size of chunks to read/write (default: 8192)
    
    Raises:
        urllib.error.URLError: If download fails
        IOError: If file cannot be written
    """
    try:
        logger.info(f"Downloading {url} to {filepath}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        # Download with progress bar
        with urllib.request.urlopen(url) as response:
            total_size = int(response.headers.get('Content-Length', 0))
            
            with open(filepath, 'wb') as f, tqdm(
                desc=os.path.basename(filepath),
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        logger.info(f"Successfully downloaded {filepath}")
    
    except urllib.error.URLError as e:
        logger.error(f"Failed to download {url}: {e}")
        raise
    except IOError as e:
        logger.error(f"Failed to write file {filepath}: {e}")
        raise


def download_dataset(data_dir: str = "data/raw") -> Dict[str, str]:
    """
    Download the Amazon co-purchasing network dataset from SNAP.
    
    Downloads both the graph file and communities file.
    
    Args:
        data_dir: Directory where data files will be saved (default: "data/raw")
    
    Returns:
        Dictionary with keys 'graph' and 'communities' containing file paths
    
    Raises:
        urllib.error.URLError: If download fails
        IOError: If files cannot be written
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    graph_path = str(data_path / "com-amazon.ungraph.txt.gz")
    communities_path = str(data_path / "com-amazon.all.cmty.txt.gz")
    
    filepaths = {}
    
    # Download graph file if it doesn't exist
    if not os.path.exists(graph_path):
        download_file(GRAPH_URL, graph_path)
    else:
        logger.info(f"Graph file already exists: {graph_path}")
    
    filepaths['graph'] = graph_path
    
    # Download communities file if it doesn't exist
    if not os.path.exists(communities_path):
        download_file(COMMUNITIES_URL, communities_path)
    else:
        logger.info(f"Communities file already exists: {communities_path}")
    
    filepaths['communities'] = communities_path
    
    return filepaths


def load_graph(filepath: str, is_gzipped: bool = True) -> nx.Graph:
    """
    Load graph from edge list file.
    
    Reads an edge list file (format: node1 node2) and creates a NetworkX
    undirected graph. Skips comment lines starting with '#'.
    
    Args:
        filepath: Path to the edge list file
        is_gzipped: Whether the file is gzip-compressed (default: True)
    
    Returns:
        NetworkX undirected graph object
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
        IOError: If file cannot be read
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Graph file not found: {filepath}")
    
    logger.info(f"Loading graph from {filepath}")
    graph = nx.Graph()
    edges_added = 0
    
    try:
        # Open file (gzipped or regular)
        if is_gzipped:
            open_func = gzip.open
            mode = 'rt'  # text mode for gzip
        else:
            open_func = open
            mode = 'r'
        
        with open_func(filepath, mode) as f:
            for line_num, line in enumerate(tqdm(f, desc="Loading edges"), start=1):
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Parse edge (format: node1 node2)
                parts = line.split()
                if len(parts) < 2:
                    logger.warning(f"Skipping invalid line {line_num}: {line}")
                    continue
                
                try:
                    node1 = int(parts[0])
                    node2 = int(parts[1])
                    graph.add_edge(node1, node2)
                    edges_added += 1
                except ValueError as e:
                    logger.warning(f"Skipping line {line_num} due to invalid node IDs: {e}")
                    continue
        
        logger.info(f"Graph loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        return graph
    
    except IOError as e:
        logger.error(f"Failed to read file {filepath}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while loading graph: {e}")
        raise ValueError(f"Failed to load graph from {filepath}: {e}")


def load_communities(filepath: str, is_gzipped: bool = True) -> Dict[int, List[int]]:
    """
    Load ground-truth communities from file.
    
    Reads communities where each line represents a community with
    space-separated node IDs.
    
    Args:
        filepath: Path to the communities file
        is_gzipped: Whether the file is gzip-compressed (default: True)
    
    Returns:
        Dictionary mapping community_id to list of node IDs
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
        IOError: If file cannot be read
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Communities file not found: {filepath}")
    
    logger.info(f"Loading communities from {filepath}")
    communities = {}
    
    try:
        # Open file (gzipped or regular)
        if is_gzipped:
            open_func = gzip.open
            mode = 'rt'  # text mode for gzip
        else:
            open_func = open
            mode = 'r'
        
        with open_func(filepath, mode) as f:
            for community_id, line in enumerate(tqdm(f, desc="Loading communities"), start=0):
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Parse community (space-separated node IDs)
                try:
                    nodes = [int(node_id) for node_id in line.split()]
                    if nodes:  # Only add non-empty communities
                        communities[community_id] = nodes
                except ValueError as e:
                    logger.warning(f"Skipping invalid community line {community_id}: {e}")
                    continue
        
        logger.info(f"Loaded {len(communities)} communities")
        return communities
    
    except IOError as e:
        logger.error(f"Failed to read file {filepath}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while loading communities: {e}")
        raise ValueError(f"Failed to load communities from {filepath}: {e}")


def save_graph(graph: nx.Graph, filepath: str) -> None:
    """
    Save NetworkX graph to file using pickle.
    
    Args:
        graph: NetworkX graph object to save
        filepath: Path where the graph will be saved
    
    Raises:
        IOError: If file cannot be written
        TypeError: If graph is not a NetworkX graph
    """
    if not isinstance(graph, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise TypeError(f"Expected NetworkX graph, got {type(graph)}")
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        logger.info(f"Saving graph to {filepath}")
        with open(filepath, 'wb') as f:
            pickle.dump(graph, f)
        
        logger.info(f"Graph saved: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    except IOError as e:
        logger.error(f"Failed to write graph to {filepath}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while saving graph: {e}")
        raise


def load_saved_graph(filepath: str) -> nx.Graph:
    """
    Load NetworkX graph from pickle file.
    
    Args:
        filepath: Path to the pickled graph file
    
    Returns:
        NetworkX graph object
    
    Raises:
        FileNotFoundError: If file doesn't exist
        IOError: If file cannot be read
        pickle.UnpicklingError: If file is not a valid pickle file
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Graph file not found: {filepath}")
    
    try:
        logger.info(f"Loading saved graph from {filepath}")
        with open(filepath, 'rb') as f:
            graph = pickle.load(f)
        
        if not isinstance(graph, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
            raise ValueError(f"File does not contain a NetworkX graph: {type(graph)}")
        
        logger.info(f"Graph loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        return graph
    
    except IOError as e:
        logger.error(f"Failed to read graph file {filepath}: {e}")
        raise
    except pickle.UnpicklingError as e:
        logger.error(f"Failed to unpickle graph from {filepath}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while loading saved graph: {e}")
        raise


def main():
    """
    Main function demonstrating usage of the data loader module.
    """
    print("=" * 60)
    print("Amazon Co-Purchasing Network Data Loader - Demo")
    print("=" * 60)
    
    # Set data directory
    data_dir = "data/raw"
    processed_dir = "data/processed"
    
    try:
        # 1. Download dataset
        print("\n1. Downloading dataset...")
        filepaths = download_dataset(data_dir)
        print(f"   Graph file: {filepaths['graph']}")
        print(f"   Communities file: {filepaths['communities']}")
        
        # 2. Load graph
        print("\n2. Loading graph...")
        graph = load_graph(filepaths['graph'], is_gzipped=True)
        print(f"   Nodes: {graph.number_of_nodes():,}")
        print(f"   Edges: {graph.number_of_edges():,}")
        print(f"   Is connected: {nx.is_connected(graph)}")
        print(f"   Number of connected components: {nx.number_connected_components(graph)}")
        
        # 3. Load communities
        print("\n3. Loading communities...")
        communities = load_communities(filepaths['communities'], is_gzipped=True)
        print(f"   Number of communities: {len(communities):,}")
        if communities:
            sample_comm_id = list(communities.keys())[0]
            print(f"   Sample community (ID {sample_comm_id}): {len(communities[sample_comm_id])} nodes")
        
        # 4. Save graph as pickle
        print("\n4. Saving graph as pickle...")
        pickle_path = os.path.join(processed_dir, "amazon_graph.pkl")
        save_graph(graph, pickle_path)
        print(f"   Saved to: {pickle_path}")
        
        # 5. Load saved graph
        print("\n5. Loading saved graph...")
        loaded_graph = load_saved_graph(pickle_path)
        print(f"   Loaded graph - Nodes: {loaded_graph.number_of_nodes():,}, "
              f"Edges: {loaded_graph.number_of_edges():,}")
        
        # Verify graphs are the same
        if graph.number_of_nodes() == loaded_graph.number_of_nodes() and \
           graph.number_of_edges() == loaded_graph.number_of_edges():
            print("   ✓ Graph verification: SUCCESS")
        else:
            print("   ✗ Graph verification: FAILED")
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)
    
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"\n✗ Error: {e}")
        raise


if __name__ == "__main__":
    main()

