"""
Spam farm generator module for network analysis.

This module implements spam farm generation based on textbook Section 5.4.1-5.4.2,
creating artificial structures to boost PageRank of target nodes.

Reference: Mining of Massive Datasets, Chapter 5.4
"""

import os
import time
import logging
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_simple_spam_farm(
    G: nx.Graph,
    target_node: Optional[Union[int, str]] = None,
    m: int = 1000,
    external_links: int = 10,
    seed: int = 42
) -> Tuple[nx.Graph, List[Union[int, str]], Union[int, str]]:
    """
    Create a simple spam farm with 1 target and m supporting pages.
    
    Structure:
    - Target links to all m supporting pages
    - All supporting pages link back to target only
    - Add external_links random edges from existing nodes to target
    
    Based on textbook Section 5.4.1.
    
    Args:
        G: NetworkX graph (will be modified)
        target_node: Existing node to boost, or None to create new target
        m: Number of supporting pages
        external_links: Number of random legitimate pages linking to target
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (modified_graph, spam_node_list, target_node_id)
    
    Raises:
        ValueError: If m < 1 or external_links < 0
    """
    if m < 1:
        raise ValueError("m must be at least 1")
    if external_links < 0:
        raise ValueError("external_links must be non-negative")
    
    start_time = time.time()
    logger.info(f"Creating simple spam farm: m={m}, external_links={external_links}")
    
    # Work with a copy to avoid modifying original
    G = G.copy()
    random.seed(seed)
    np.random.seed(seed)
    
    # Get max node ID for new node generation
    if G.number_of_nodes() > 0:
        if isinstance(list(G.nodes())[0], int):
            max_node_id = max(G.nodes())
            new_node_start = max_node_id + 1
        else:
            # String nodes - use numeric suffix
            numeric_nodes = [int(n) for n in G.nodes() if str(n).isdigit()]
            if numeric_nodes:
                max_node_id = max(numeric_nodes)
                new_node_start = max_node_id + 1
            else:
                new_node_start = 0
    else:
        new_node_start = 0
    
    # Create or use target node
    if target_node is None:
        target_node = new_node_start
        G.add_node(target_node)
        logger.info(f"Created new target node: {target_node}")
    elif target_node not in G:
        G.add_node(target_node)
        logger.info(f"Added target node: {target_node}")
    else:
        logger.info(f"Using existing target node: {target_node}")
    
    # Create m supporting pages
    spam_nodes = []
    for i in range(m):
        spam_node = new_node_start + 1 + i
        spam_nodes.append(spam_node)
        G.add_node(spam_node)
    
    logger.info(f"Created {m} supporting pages: nodes {spam_nodes[0]} to {spam_nodes[-1]}")
    
    # Create spam farm structure
    # Target links to all supporting pages
    for spam_node in spam_nodes:
        G.add_edge(target_node, spam_node)
    
    # All supporting pages link back to target only
    for spam_node in spam_nodes:
        G.add_edge(spam_node, target_node)
    
    logger.info(f"Created {2 * m} spam farm edges (target <-> supporting pages)")
    
    # Add external links from legitimate nodes to target
    if external_links > 0 and G.number_of_nodes() > len(spam_nodes) + 1:
        # Get legitimate nodes (not spam nodes or target)
        legitimate_nodes = [n for n in G.nodes() 
                           if n != target_node and n not in spam_nodes]
        
        if len(legitimate_nodes) >= external_links:
            external_sources = random.sample(legitimate_nodes, external_links)
            for source in external_sources:
                G.add_edge(source, target_node)
            logger.info(f"Added {external_links} external links to target")
        else:
            logger.warning(f"Not enough legitimate nodes for {external_links} external links")
    
    computation_time = time.time() - start_time
    logger.info(f"Simple spam farm created in {computation_time:.4f} seconds")
    logger.info(f"  Target node: {target_node}")
    logger.info(f"  Supporting pages: {m}")
    logger.info(f"  Total spam nodes: {len(spam_nodes)}")
    logger.info(f"  Graph size: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    return G, spam_nodes, target_node


def create_multiple_spam_farms(
    G: nx.Graph,
    num_farms: int = 5,
    m_per_farm: int = 500,
    external_links: int = 5,
    seed: int = 42
) -> Tuple[nx.Graph, Dict[int, Tuple[List[Union[int, str]], Union[int, str]]]]:
    """
    Create multiple independent spam farms.
    
    Each farm has its own target page and m supporting pages.
    No links between different spam farms.
    
    Args:
        G: NetworkX graph (will be modified)
        num_farms: Number of independent spam farms to create
        m_per_farm: Number of supporting pages per farm
        external_links: Number of external links per target
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (modified_graph, farms_dict) where farms_dict maps
        farm_id to (spam_nodes_list, target_node)
    
    Raises:
        ValueError: If num_farms < 1 or m_per_farm < 1
    """
    if num_farms < 1:
        raise ValueError("num_farms must be at least 1")
    if m_per_farm < 1:
        raise ValueError("m_per_farm must be at least 1")
    
    start_time = time.time()
    logger.info(f"Creating {num_farms} independent spam farms (m={m_per_farm} each)")
    
    G = G.copy()
    random.seed(seed)
    np.random.seed(seed)
    
    # Get starting node ID
    if G.number_of_nodes() > 0:
        if isinstance(list(G.nodes())[0], int):
            max_node_id = max(G.nodes())
            new_node_start = max_node_id + 1
        else:
            numeric_nodes = [int(n) for n in G.nodes() if str(n).isdigit()]
            if numeric_nodes:
                max_node_id = max(numeric_nodes)
                new_node_start = max_node_id + 1
            else:
                new_node_start = 0
    else:
        new_node_start = 0
    
    farms_dict = {}
    current_node_id = new_node_start
    
    for farm_id in range(num_farms):
        # Create target node
        target_node = current_node_id
        G.add_node(target_node)
        current_node_id += 1
        
        # Create supporting pages
        spam_nodes = []
        for i in range(m_per_farm):
            spam_node = current_node_id
            spam_nodes.append(spam_node)
            G.add_node(spam_node)
            current_node_id += 1
        
        # Create farm structure
        for spam_node in spam_nodes:
            G.add_edge(target_node, spam_node)
            G.add_edge(spam_node, target_node)
        
        # Add external links
        if external_links > 0:
            legitimate_nodes = [n for n in G.nodes() 
                               if n != target_node and n not in spam_nodes
                               and n not in [t for _, (_, t) in farms_dict.items()]]
            
            if len(legitimate_nodes) >= external_links:
                external_sources = random.sample(legitimate_nodes, external_links)
                for source in external_sources:
                    G.add_edge(source, target_node)
        
        farms_dict[farm_id] = (spam_nodes, target_node)
        logger.info(f"Created farm {farm_id}: target={target_node}, {m_per_farm} supporting pages")
    
    computation_time = time.time() - start_time
    logger.info(f"Created {num_farms} independent spam farms in {computation_time:.4f} seconds")
    logger.info(f"  Total spam nodes: {num_farms * m_per_farm}")
    logger.info(f"  Graph size: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    return G, farms_dict


def create_collaborative_spam_farms(
    G: nx.Graph,
    num_farms: int = 2,
    m_per_farm: int = 500,
    external_links: int = 10,
    seed: int = 42
) -> Tuple[nx.Graph, Dict[int, Tuple[List[Union[int, str]], Union[int, str]]], List[Union[int, str]]]:
    """
    Create collaborative spam farms that link to each other's targets.
    
    Based on Exercise 5.4.3 - farms boost each other by linking targets together.
    Each farm's target links to other farms' targets.
    
    Args:
        G: NetworkX graph (will be modified)
        num_farms: Number of collaborative spam farms
        m_per_farm: Number of supporting pages per farm
        external_links: Number of external links per target
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (modified_graph, farms_dict, target_nodes_list)
    
    Raises:
        ValueError: If num_farms < 2 (collaboration requires at least 2 farms)
    """
    if num_farms < 2:
        raise ValueError("num_farms must be at least 2 for collaborative farms")
    if m_per_farm < 1:
        raise ValueError("m_per_farm must be at least 1")
    
    start_time = time.time()
    logger.info(f"Creating {num_farms} collaborative spam farms (m={m_per_farm} each)")
    
    G = G.copy()
    random.seed(seed)
    np.random.seed(seed)
    
    # Get starting node ID
    if G.number_of_nodes() > 0:
        if isinstance(list(G.nodes())[0], int):
            max_node_id = max(G.nodes())
            new_node_start = max_node_id + 1
        else:
            numeric_nodes = [int(n) for n in G.nodes() if str(n).isdigit()]
            if numeric_nodes:
                max_node_id = max(numeric_nodes)
                new_node_start = max_node_id + 1
            else:
                new_node_start = 0
    else:
        new_node_start = 0
    
    farms_dict = {}
    target_nodes = []
    current_node_id = new_node_start
    
    # Create all farms first
    for farm_id in range(num_farms):
        # Create target node
        target_node = current_node_id
        G.add_node(target_node)
        target_nodes.append(target_node)
        current_node_id += 1
        
        # Create supporting pages
        spam_nodes = []
        for i in range(m_per_farm):
            spam_node = current_node_id
            spam_nodes.append(spam_node)
            G.add_node(spam_node)
            current_node_id += 1
        
        # Create farm structure
        for spam_node in spam_nodes:
            G.add_edge(target_node, spam_node)
            G.add_edge(spam_node, target_node)
        
        farms_dict[farm_id] = (spam_nodes, target_node)
    
    # Add cross-links between targets (collaboration)
    for i, target_i in enumerate(target_nodes):
        for j, target_j in enumerate(target_nodes):
            if i != j:
                G.add_edge(target_i, target_j)
    
    logger.info(f"Added {num_farms * (num_farms - 1)} cross-links between targets")
    
    # Add external links to each target
    for farm_id, (spam_nodes, target_node) in farms_dict.items():
        if external_links > 0:
            # Get all legitimate nodes (not in any spam farm)
            all_spam_nodes = set()
            all_targets = set(target_nodes)
            for _, (sn, _) in farms_dict.items():
                all_spam_nodes.update(sn)
            
            legitimate_nodes = [n for n in G.nodes() 
                               if n not in all_spam_nodes and n not in all_targets]
            
            if len(legitimate_nodes) >= external_links:
                external_sources = random.sample(legitimate_nodes, external_links)
                for source in external_sources:
                    G.add_edge(source, target_node)
    
    computation_time = time.time() - start_time
    logger.info(f"Created {num_farms} collaborative spam farms in {computation_time:.4f} seconds")
    logger.info(f"  Total spam nodes: {num_farms * m_per_farm}")
    logger.info(f"  Target nodes: {target_nodes}")
    logger.info(f"  Graph size: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    return G, farms_dict, target_nodes


def inject_spam_into_network(
    G: nx.Graph,
    spam_type: str = 'simple',
    **kwargs
) -> Tuple[nx.Graph, Dict]:
    """
    Wrapper function to inject spam into network.
    
    Args:
        G: NetworkX graph
        spam_type: Type of spam farm ('simple', 'multiple', 'collaborative')
        **kwargs: Additional arguments passed to spam farm creator
    
    Returns:
        Tuple of (modified_graph, spam_metadata_dict)
    
    Raises:
        ValueError: If spam_type is unknown
    """
    logger.info(f"Injecting spam into network: type={spam_type}")
    
    start_time = time.time()
    
    if spam_type == 'simple':
        G_modified, spam_nodes, target_node = create_simple_spam_farm(G, **kwargs)
        metadata = {
            'spam_type': 'simple',
            'spam_nodes': spam_nodes,
            'target_node': target_node,
            'num_supporting_pages': len(spam_nodes),
            'external_links': kwargs.get('external_links', 10)
        }
    
    elif spam_type == 'multiple':
        G_modified, farms_dict = create_multiple_spam_farms(G, **kwargs)
        all_spam_nodes = []
        all_targets = []
        for farm_id, (spam_nodes, target) in farms_dict.items():
            all_spam_nodes.extend(spam_nodes)
            all_targets.append(target)
        
        metadata = {
            'spam_type': 'multiple',
            'farms_dict': {str(k): {'spam_nodes': v[0], 'target': v[1]} 
                          for k, v in farms_dict.items()},
            'all_spam_nodes': all_spam_nodes,
            'all_target_nodes': all_targets,
            'num_farms': len(farms_dict),
            'm_per_farm': kwargs.get('m_per_farm', 500)
        }
    
    elif spam_type == 'collaborative':
        G_modified, farms_dict, target_nodes = create_collaborative_spam_farms(G, **kwargs)
        all_spam_nodes = []
        for spam_nodes, _ in farms_dict.values():
            all_spam_nodes.extend(spam_nodes)
        
        metadata = {
            'spam_type': 'collaborative',
            'farms_dict': {str(k): {'spam_nodes': v[0], 'target': v[1]} 
                          for k, v in farms_dict.items()},
            'all_spam_nodes': all_spam_nodes,
            'all_target_nodes': target_nodes,
            'num_farms': len(farms_dict),
            'm_per_farm': kwargs.get('m_per_farm', 500),
            'has_cross_links': True
        }
    
    else:
        raise ValueError(f"Unknown spam_type: {spam_type}. Must be 'simple', 'multiple', or 'collaborative'")
    
    # Add general metadata
    metadata['timestamp'] = datetime.now().isoformat()
    metadata['original_nodes'] = G.number_of_nodes()
    metadata['original_edges'] = G.number_of_edges()
    metadata['modified_nodes'] = G_modified.number_of_nodes()
    metadata['modified_edges'] = G_modified.number_of_edges()
    metadata['injection_time'] = time.time() - start_time
    
    logger.info(f"Spam injection complete: {metadata['injection_time']:.4f} seconds")
    
    return G_modified, metadata


def calculate_theoretical_pagerank_boost(
    m: int,
    n: int,
    beta: float = 0.85,
    x: float = 0.01
) -> float:
    """
    Calculate theoretical PageRank boost for spam farm target.
    
    Based on textbook formula (Section 5.4.2):
    y = x/(1-β²) + c*m/n
    
    where c = β/(1+β)
    
    Args:
        m: Number of supporting pages
        n: Total number of nodes in network
        beta: Damping factor (default: 0.85)
        x: External PageRank contribution (default: 0.01)
    
    Returns:
        Expected PageRank value for target node
    
    Raises:
        ValueError: If parameters are invalid
    """
    if m < 1:
        raise ValueError("m must be at least 1")
    if n < 1:
        raise ValueError("n must be at least 1")
    if not 0 < beta < 1:
        raise ValueError("beta must be between 0 and 1")
    if x < 0:
        raise ValueError("x must be non-negative")
    
    # Calculate c = β/(1+β)
    c = beta / (1 + beta)
    
    # Calculate y = x/(1-β²) + c*m/n
    y = x / (1 - beta**2) + c * m / n
    
    logger.debug(f"Theoretical PageRank boost: m={m}, n={n}, beta={beta}, x={x} -> y={y:.6f}")
    
    return y


def save_spam_metadata(
    spam_info: Dict,
    output_path: str
) -> None:
    """
    Save spam metadata to JSON file.
    
    Args:
        spam_info: Dictionary containing spam metadata
        output_path: Path to save JSON file
    
    Raises:
        ValueError: If spam_info is empty
    """
    if not spam_info:
        raise ValueError("spam_info is empty")
    
    logger.info(f"Saving spam metadata to {output_path}")
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Convert any non-serializable types
    json_info = {}
    for key, value in spam_info.items():
        if isinstance(value, (list, tuple)):
            json_info[key] = [int(v) if isinstance(v, (int, np.integer)) else str(v) for v in value]
        elif isinstance(value, dict):
            json_info[key] = {
                k: [int(vv) if isinstance(vv, (int, np.integer)) else str(vv) for vv in v] 
                if isinstance(v, (list, tuple)) else int(v) if isinstance(v, (int, np.integer)) else str(v)
                for k, v in value.items()
            }
        elif isinstance(value, (int, np.integer)):
            json_info[key] = int(value)
        elif isinstance(value, (float, np.floating)):
            json_info[key] = float(value)
        else:
            json_info[key] = value
    
    with open(output_path, 'w') as f:
        json.dump(json_info, f, indent=2)
    
    logger.info(f"Spam metadata saved to {output_path}")


def visualize_spam_farm(
    G: nx.Graph,
    spam_nodes: List[Union[int, str]],
    target_node: Union[int, str],
    save_path: Optional[str] = None,
    external_nodes: Optional[List[Union[int, str]]] = None,
    figsize: Tuple[int, int] = (14, 10),
    max_nodes_to_plot: int = 200
) -> None:
    """
    Visualize spam farm structure with color coding.
    
    Args:
        G: NetworkX graph
        spam_nodes: List of spam node IDs
        target_node: Target node ID
        save_path: Optional path to save figure
        external_nodes: Optional list of external nodes linking to target
        figsize: Figure size tuple (default: (14, 10))
        max_nodes_to_plot: Maximum nodes to include in visualization (default: 200)
    
    Raises:
        ValueError: If target_node not in graph
    """
    if target_node not in G:
        raise ValueError(f"Target node {target_node} not in graph")
    
    logger.info(f"Visualizing spam farm: target={target_node}, {len(spam_nodes)} spam nodes")
    
    # Create subgraph with spam farm and some neighbors
    nodes_to_plot = {target_node}
    nodes_to_plot.update(spam_nodes)
    
    if external_nodes:
        nodes_to_plot.update(external_nodes[:max_nodes_to_plot - len(nodes_to_plot)])
    
    # Add neighbors of target (external links)
    if len(nodes_to_plot) < max_nodes_to_plot:
        target_neighbors = list(G.neighbors(target_node))
        remaining = max_nodes_to_plot - len(nodes_to_plot)
        nodes_to_plot.update(target_neighbors[:remaining])
    
    G_sub = G.subgraph(list(nodes_to_plot)).copy()
    
    # Compute layout
    pos = nx.spring_layout(G_sub, k=1, iterations=50, seed=42)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Identify node types
    spam_node_set = set(spam_nodes)
    external_node_set = set(external_nodes) if external_nodes else set()
    external_node_set = external_node_set - {target_node} - spam_node_set
    
    # Draw edges
    nx.draw_networkx_edges(G_sub, pos, alpha=0.2, width=0.5, ax=ax, edge_color='gray')
    
    # Draw external nodes
    if external_node_set:
        external_plot = [n for n in G_sub.nodes() if n in external_node_set]
        if external_plot:
            nx.draw_networkx_nodes(
                G_sub, pos,
                nodelist=external_plot,
                node_color='lightblue',
                node_size=100,
                alpha=0.6,
                ax=ax,
                label='External Nodes'
            )
    
    # Draw spam nodes
    spam_plot = [n for n in G_sub.nodes() if n in spam_node_set]
    if spam_plot:
        nx.draw_networkx_nodes(
            G_sub, pos,
            nodelist=spam_plot,
            node_color='orange',
            node_size=200,
            alpha=0.7,
            ax=ax,
            label='Spam Nodes'
        )
    
    # Draw target node
    nx.draw_networkx_nodes(
        G_sub, pos,
        nodelist=[target_node],
        node_color='red',
        node_size=1000,
        alpha=1.0,
        ax=ax,
        label='Target Node'
    )
    
    # Draw labels
    labels = {target_node: f'T{target_node}'}
    nx.draw_networkx_labels(G_sub, pos, labels, font_size=10, font_weight='bold', ax=ax)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=10)
    
    # Remove axes
    ax.axis('off')
    ax.set_title(f'Spam Farm Visualization\nTarget: {target_node}, Spam Nodes: {len(spam_nodes)}',
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Spam farm visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """
    Main function demonstrating spam farm generation.
    """
    print("=" * 60)
    print("Spam Farm Generator - Demo")
    print("=" * 60)
    
    try:
        import networkx as nx
        
        # Create a sample graph
        print("\n1. Creating sample graph...")
        G = nx.erdos_renyi_graph(n=1000, p=0.01, seed=42)
        print(f"   Created graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Test simple spam farm
        print("\n2. Creating simple spam farm (m=100)...")
        G_spam, spam_nodes, target = create_simple_spam_farm(
            G, target_node=None, m=100, external_links=10, seed=42
        )
        print(f"   Target node: {target}")
        print(f"   Spam nodes: {len(spam_nodes)}")
        print(f"   Modified graph: {G_spam.number_of_nodes()} nodes, {G_spam.number_of_edges()} edges")
        
        # Calculate theoretical boost
        print("\n3. Calculating theoretical PageRank boost...")
        theoretical_pr = calculate_theoretical_pagerank_boost(
            m=100, n=G_spam.number_of_nodes(), beta=0.85, x=0.01
        )
        print(f"   Theoretical PageRank: {theoretical_pr:.6f}")
        
        # Visualize
        print("\n4. Visualizing spam farm...")
        visualize_spam_farm(
            G_spam, spam_nodes, target,
            save_path='results/figures/spam_farm_simple.png',
            max_nodes_to_plot=150
        )
        print("   ✅ Visualization saved")
        
        # Save metadata
        print("\n5. Saving spam metadata...")
        metadata = {
            'spam_type': 'simple',
            'spam_nodes': spam_nodes,
            'target_node': target,
            'num_supporting_pages': len(spam_nodes),
            'theoretical_pagerank': theoretical_pr
        }
        save_spam_metadata(metadata, 'results/spam_metadata_simple.json')
        print("   ✅ Metadata saved")
        
        # Test multiple farms
        print("\n6. Creating multiple spam farms...")
        G_multi, farms_dict = create_multiple_spam_farms(
            G, num_farms=3, m_per_farm=50, external_links=5, seed=42
        )
        print(f"   Created {len(farms_dict)} independent farms")
        print(f"   Modified graph: {G_multi.number_of_nodes()} nodes, {G_multi.number_of_edges()} edges")
        
        # Test collaborative farms
        print("\n7. Creating collaborative spam farms...")
        G_collab, collab_farms, targets = create_collaborative_spam_farms(
            G, num_farms=2, m_per_farm=50, external_links=10, seed=42
        )
        print(f"   Created {len(collab_farms)} collaborative farms")
        print(f"   Target nodes: {targets}")
        print(f"   Modified graph: {G_collab.number_of_nodes()} nodes, {G_collab.number_of_edges()} edges")
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)
    
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"\n✗ Error: {e}")
        raise


if __name__ == "__main__":
    main()

