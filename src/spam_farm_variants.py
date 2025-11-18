"""
Spam farm variants module for network analysis.

This module implements various spam farm configurations based on textbook
exercises and variations to analyze their effectiveness.

Reference: Mining of Massive Datasets, Chapter 5.4 Exercises
"""

import os
import time
import logging
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import random

# Import base spam farm functions
from spam_farm_generator import (
    calculate_theoretical_pagerank_boost,
    save_spam_metadata
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_self_linking_spam_farm(
    G: nx.Graph,
    target_node: Optional[Union[int, str]] = None,
    m: int = 1000,
    external_links: int = 10,
    seed: int = 42
) -> Tuple[nx.Graph, Dict]:
    """
    Create spam farm where supporting pages link to THEMSELVES instead of target.
    
    Exercise 5.4.1(a): Each supporting page links to itself (self-loop).
    This variant tests how self-linking affects PageRank boost.
    
    Structure:
    - Target links to all m supporting pages
    - Each supporting page links to ITSELF (self-loop)
    - Add external_links random edges from existing nodes to target
    
    Args:
        G: NetworkX graph (will be modified)
        target_node: Existing node to boost, or None to create new target
        m: Number of supporting pages
        external_links: Number of random legitimate pages linking to target
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (modified_graph, spam_metadata_dict)
    
    Raises:
        ValueError: If m < 1 or external_links < 0
    """
    if m < 1:
        raise ValueError("m must be at least 1")
    if external_links < 0:
        raise ValueError("external_links must be non-negative")
    
    start_time = time.time()
    logger.info(f"Creating self-linking spam farm: m={m}, external_links={external_links}")
    logger.info("Exercise 5.4.1(a): Supporting pages link to themselves")
    
    G = G.copy()
    random.seed(seed)
    np.random.seed(seed)
    
    # Get max node ID
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
    
    # Create or use target node
    if target_node is None:
        target_node = new_node_start
        G.add_node(target_node)
    elif target_node not in G:
        G.add_node(target_node)
    
    # Create m supporting pages
    spam_nodes = []
    for i in range(m):
        spam_node = new_node_start + 1 + i
        spam_nodes.append(spam_node)
        G.add_node(spam_node)
    
    # Create spam farm structure
    # Target links to all supporting pages
    for spam_node in spam_nodes:
        G.add_edge(target_node, spam_node)
    
    # Each supporting page links to ITSELF (self-loop)
    for spam_node in spam_nodes:
        G.add_edge(spam_node, spam_node)
    
    logger.info(f"Created {m} self-loops on supporting pages")
    
    # Add external links
    if external_links > 0 and G.number_of_nodes() > len(spam_nodes) + 1:
        legitimate_nodes = [n for n in G.nodes() 
                           if n != target_node and n not in spam_nodes]
        if len(legitimate_nodes) >= external_links:
            external_sources = random.sample(legitimate_nodes, external_links)
            for source in external_sources:
                G.add_edge(source, target_node)
    
    computation_time = time.time() - start_time
    
    metadata = {
        'spam_type': 'self_linking',
        'spam_nodes': spam_nodes,
        'target_node': target_node,
        'num_supporting_pages': m,
        'external_links': external_links,
        'num_self_loops': m,
        'timestamp': datetime.now().isoformat(),
        'computation_time': computation_time
    }
    
    logger.info(f"Self-linking spam farm created in {computation_time:.4f} seconds")
    
    return G, metadata


def create_dead_end_spam_farm(
    G: nx.Graph,
    target_node: Optional[Union[int, str]] = None,
    m: int = 1000,
    external_links: int = 10,
    seed: int = 42
) -> Tuple[nx.Graph, Dict]:
    """
    Create spam farm where supporting pages are dead ends (link to nowhere).
    
    Exercise 5.4.1(b): Supporting pages link to NOWHERE (dead ends).
    This variant tests how dead ends affect PageRank boost.
    
    Structure:
    - Target links to all m supporting pages
    - Supporting pages link to NOTHING (dead ends)
    - Add external_links random edges from existing nodes to target
    
    Args:
        G: NetworkX graph (will be modified)
        target_node: Existing node to boost, or None to create new target
        m: Number of supporting pages
        external_links: Number of random legitimate pages linking to target
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (modified_graph, spam_metadata_dict)
    
    Raises:
        ValueError: If m < 1 or external_links < 0
    """
    if m < 1:
        raise ValueError("m must be at least 1")
    if external_links < 0:
        raise ValueError("external_links must be non-negative")
    
    start_time = time.time()
    logger.info(f"Creating dead-end spam farm: m={m}, external_links={external_links}")
    logger.info("Exercise 5.4.1(b): Supporting pages are dead ends")
    
    G = G.copy()
    random.seed(seed)
    np.random.seed(seed)
    
    # Get max node ID
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
    
    # Create or use target node
    if target_node is None:
        target_node = new_node_start
        G.add_node(target_node)
    elif target_node not in G:
        G.add_node(target_node)
    
    # Create m supporting pages (dead ends)
    spam_nodes = []
    for i in range(m):
        spam_node = new_node_start + 1 + i
        spam_nodes.append(spam_node)
        G.add_node(spam_node)
    
    # Create spam farm structure
    # Target links to all supporting pages
    for spam_node in spam_nodes:
        G.add_edge(target_node, spam_node)
    
    # Supporting pages link to NOTHING (dead ends)
    # No edges added from supporting pages
    
    logger.info(f"Created {m} dead-end supporting pages (no outgoing links)")
    
    # Add external links
    if external_links > 0 and G.number_of_nodes() > len(spam_nodes) + 1:
        legitimate_nodes = [n for n in G.nodes() 
                           if n != target_node and n not in spam_nodes]
        if len(legitimate_nodes) >= external_links:
            external_sources = random.sample(legitimate_nodes, external_links)
            for source in external_sources:
                G.add_edge(source, target_node)
    
    computation_time = time.time() - start_time
    
    metadata = {
        'spam_type': 'dead_end',
        'spam_nodes': spam_nodes,
        'target_node': target_node,
        'num_supporting_pages': m,
        'external_links': external_links,
        'num_dead_ends': m,
        'timestamp': datetime.now().isoformat(),
        'computation_time': computation_time
    }
    
    logger.info(f"Dead-end spam farm created in {computation_time:.4f} seconds")
    
    return G, metadata


def create_dual_link_spam_farm(
    G: nx.Graph,
    target_node: Optional[Union[int, str]] = None,
    m: int = 1000,
    external_links: int = 10,
    seed: int = 42
) -> Tuple[nx.Graph, Dict]:
    """
    Create spam farm where supporting pages link to BOTH themselves AND target.
    
    Exercise 5.4.1(c): Each supporting page links to both itself and target.
    This variant tests dual-linking effectiveness.
    
    Structure:
    - Target links to all m supporting pages
    - Each supporting page links to ITSELF (self-loop)
    - Each supporting page ALSO links to TARGET
    - Add external_links random edges from existing nodes to target
    
    Args:
        G: NetworkX graph (will be modified)
        target_node: Existing node to boost, or None to create new target
        m: Number of supporting pages
        external_links: Number of random legitimate pages linking to target
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (modified_graph, spam_metadata_dict)
    
    Raises:
        ValueError: If m < 1 or external_links < 0
    """
    if m < 1:
        raise ValueError("m must be at least 1")
    if external_links < 0:
        raise ValueError("external_links must be non-negative")
    
    start_time = time.time()
    logger.info(f"Creating dual-link spam farm: m={m}, external_links={external_links}")
    logger.info("Exercise 5.4.1(c): Supporting pages link to both themselves and target")
    
    G = G.copy()
    random.seed(seed)
    np.random.seed(seed)
    
    # Get max node ID
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
    
    # Create or use target node
    if target_node is None:
        target_node = new_node_start
        G.add_node(target_node)
    elif target_node not in G:
        G.add_node(target_node)
    
    # Create m supporting pages
    spam_nodes = []
    for i in range(m):
        spam_node = new_node_start + 1 + i
        spam_nodes.append(spam_node)
        G.add_node(spam_node)
    
    # Create spam farm structure
    # Target links to all supporting pages
    for spam_node in spam_nodes:
        G.add_edge(target_node, spam_node)
    
    # Each supporting page links to BOTH itself AND target
    for spam_node in spam_nodes:
        G.add_edge(spam_node, spam_node)  # Self-loop
        G.add_edge(spam_node, target_node)  # Link to target
    
    logger.info(f"Created {m} dual-links (self + target) per supporting page")
    
    # Add external links
    if external_links > 0 and G.number_of_nodes() > len(spam_nodes) + 1:
        legitimate_nodes = [n for n in G.nodes() 
                           if n != target_node and n not in spam_nodes]
        if len(legitimate_nodes) >= external_links:
            external_sources = random.sample(legitimate_nodes, external_links)
            for source in external_sources:
                G.add_edge(source, target_node)
    
    computation_time = time.time() - start_time
    
    metadata = {
        'spam_type': 'dual_link',
        'spam_nodes': spam_nodes,
        'target_node': target_node,
        'num_supporting_pages': m,
        'external_links': external_links,
        'num_self_loops': m,
        'num_target_links': m,
        'timestamp': datetime.now().isoformat(),
        'computation_time': computation_time
    }
    
    logger.info(f"Dual-link spam farm created in {computation_time:.4f} seconds")
    
    return G, metadata


def create_honeypot_spam_farm(
    G: nx.Graph,
    target_node: Optional[Union[int, str]] = None,
    m: int = 1000,
    external_links: int = 100,
    seed: int = 42
) -> Tuple[nx.Graph, Dict]:
    """
    Create honeypot spam farm with many external links to target.
    
    Honeypot strategy: Attract many legitimate links to target, then
    redirect PageRank to supporting pages.
    
    Structure:
    - Many external links TO target (honeypot)
    - Target links to many supporting pages
    - Supporting pages form tight cluster (link to each other)
    - Supporting pages link back to target
    
    Args:
        G: NetworkX graph (will be modified)
        target_node: Existing node to boost, or None to create new target
        m: Number of supporting pages
        external_links: Number of external links (higher than normal)
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (modified_graph, spam_metadata_dict)
    """
    if m < 1:
        raise ValueError("m must be at least 1")
    if external_links < 0:
        raise ValueError("external_links must be non-negative")
    
    start_time = time.time()
    logger.info(f"Creating honeypot spam farm: m={m}, external_links={external_links}")
    
    G = G.copy()
    random.seed(seed)
    np.random.seed(seed)
    
    # Get max node ID
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
    
    # Create or use target node
    if target_node is None:
        target_node = new_node_start
        G.add_node(target_node)
    elif target_node not in G:
        G.add_node(target_node)
    
    # Create m supporting pages
    spam_nodes = []
    for i in range(m):
        spam_node = new_node_start + 1 + i
        spam_nodes.append(spam_node)
        G.add_node(spam_node)
    
    # Create spam farm structure
    # Target links to all supporting pages
    for spam_node in spam_nodes:
        G.add_edge(target_node, spam_node)
    
    # Supporting pages link back to target
    for spam_node in spam_nodes:
        G.add_edge(spam_node, target_node)
    
    # Supporting pages form tight cluster (link to each other)
    # Add some random links between supporting pages
    cluster_links = min(m * 2, len(spam_nodes) * (len(spam_nodes) - 1) // 4)
    for _ in range(cluster_links):
        u, v = random.sample(spam_nodes, 2)
        if u != v:
            G.add_edge(u, v)
    
    logger.info(f"Created {cluster_links} cluster links between supporting pages")
    
    # Add many external links (honeypot)
    if external_links > 0 and G.number_of_nodes() > len(spam_nodes) + 1:
        legitimate_nodes = [n for n in G.nodes() 
                           if n != target_node and n not in spam_nodes]
        if len(legitimate_nodes) >= external_links:
            external_sources = random.sample(legitimate_nodes, external_links)
            for source in external_sources:
                G.add_edge(source, target_node)
        else:
            # Use all available legitimate nodes
            for source in legitimate_nodes:
                G.add_edge(source, target_node)
            external_links = len(legitimate_nodes)
    
    computation_time = time.time() - start_time
    
    metadata = {
        'spam_type': 'honeypot',
        'spam_nodes': spam_nodes,
        'target_node': target_node,
        'num_supporting_pages': m,
        'external_links': external_links,
        'cluster_links': cluster_links,
        'timestamp': datetime.now().isoformat(),
        'computation_time': computation_time
    }
    
    logger.info(f"Honeypot spam farm created in {computation_time:.4f} seconds")
    
    return G, metadata


def create_layered_spam_farm(
    G: nx.Graph,
    target_node: Optional[Union[int, str]] = None,
    layers: int = 3,
    nodes_per_layer: int = 100,
    external_links: int = 10,
    seed: int = 42
) -> Tuple[nx.Graph, Dict]:
    """
    Create layered (pyramid) spam farm structure.
    
    Pyramid structure with multiple layers:
    - Layer 1 (closest to target) links to target
    - Layer 2 links to Layer 1
    - Layer 3 links to Layer 2, etc.
    
    This creates a hierarchical structure to boost target PageRank.
    
    Args:
        G: NetworkX graph (will be modified)
        target_node: Existing node to boost, or None to create new target
        layers: Number of layers in pyramid
        nodes_per_layer: Number of nodes per layer
        external_links: Number of external links to target
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (modified_graph, spam_metadata_dict)
    """
    if layers < 1:
        raise ValueError("layers must be at least 1")
    if nodes_per_layer < 1:
        raise ValueError("nodes_per_layer must be at least 1")
    
    start_time = time.time()
    logger.info(f"Creating layered spam farm: layers={layers}, nodes_per_layer={nodes_per_layer}")
    
    G = G.copy()
    random.seed(seed)
    np.random.seed(seed)
    
    # Get max node ID
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
    
    # Create or use target node
    if target_node is None:
        target_node = new_node_start
        G.add_node(target_node)
    elif target_node not in G:
        G.add_node(target_node)
    
    current_node_id = new_node_start + 1
    all_spam_nodes = []
    layer_nodes = {}
    
    # Create layers
    for layer in range(1, layers + 1):
        layer_node_list = []
        for i in range(nodes_per_layer):
            node_id = current_node_id
            layer_node_list.append(node_id)
            all_spam_nodes.append(node_id)
            G.add_node(node_id)
            current_node_id += 1
        
        layer_nodes[layer] = layer_node_list
        
        # Link layer to previous layer or target
        if layer == 1:
            # Layer 1 links to target
            for node in layer_node_list:
                G.add_edge(node, target_node)
                G.add_edge(target_node, node)  # Bidirectional
        else:
            # Link to previous layer
            prev_layer_nodes = layer_nodes[layer - 1]
            for node in layer_node_list:
                # Link to some nodes in previous layer
                num_links = min(5, len(prev_layer_nodes))
                targets = random.sample(prev_layer_nodes, num_links)
                for target in targets:
                    G.add_edge(node, target)
    
    logger.info(f"Created {layers} layers with {nodes_per_layer} nodes each")
    
    # Add external links
    if external_links > 0 and G.number_of_nodes() > len(all_spam_nodes) + 1:
        legitimate_nodes = [n for n in G.nodes() 
                           if n != target_node and n not in all_spam_nodes]
        if len(legitimate_nodes) >= external_links:
            external_sources = random.sample(legitimate_nodes, external_links)
            for source in external_sources:
                G.add_edge(source, target_node)
    
    computation_time = time.time() - start_time
    
    metadata = {
        'spam_type': 'layered',
        'spam_nodes': all_spam_nodes,
        'target_node': target_node,
        'num_layers': layers,
        'nodes_per_layer': nodes_per_layer,
        'total_spam_nodes': len(all_spam_nodes),
        'external_links': external_links,
        'layer_structure': {str(k): len(v) for k, v in layer_nodes.items()},
        'timestamp': datetime.now().isoformat(),
        'computation_time': computation_time
    }
    
    logger.info(f"Layered spam farm created in {computation_time:.4f} seconds")
    
    return G, metadata


def experiment_spam_parameters(
    G_original: nx.Graph,
    m_values: List[int] = [100, 500, 1000, 5000],
    external_values: List[int] = [5, 10, 50, 100],
    beta_values: List[float] = [0.8, 0.85, 0.9],
    spam_type: str = 'simple',
    seed: int = 42
) -> pd.DataFrame:
    """
    Run experiments with different spam farm parameters.
    
    For each combination of parameters:
    - Create spam farm
    - Compute PageRank
    - Calculate boost factor
    - Compare with theoretical prediction
    
    Args:
        G_original: Original graph (will be copied for each experiment)
        m_values: List of m (supporting pages) values to test
        external_values: List of external link counts to test
        beta_values: List of beta (damping factor) values to test
        spam_type: Type of spam farm ('simple', 'self_linking', etc.)
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with columns: m, external_links, beta, target_pagerank,
        theoretical_pagerank, boost_factor, error
    """
    logger.info(f"Running spam parameter experiments: {len(m_values)} m values, "
               f"{len(external_values)} external values, {len(beta_values)} beta values")
    
    results = []
    total_experiments = len(m_values) * len(external_values) * len(beta_values)
    experiment_num = 0
    
    from spam_farm_generator import create_simple_spam_farm
    from centrality_analysis import compute_pagerank
    
    for m in m_values:
        for external in external_values:
            for beta in beta_values:
                experiment_num += 1
                logger.info(f"Experiment {experiment_num}/{total_experiments}: m={m}, external={external}, beta={beta}")
                
                try:
                    # Create spam farm
                    if spam_type == 'simple':
                        G_spam, spam_nodes, target = create_simple_spam_farm(
                            G_original, m=m, external_links=external, seed=seed + experiment_num
                        )
                    else:
                        # For other types, use inject function
                        from spam_farm_generator import inject_spam_into_network
                        G_spam, metadata = inject_spam_into_network(
                            G_original, spam_type=spam_type, m=m, external_links=external, seed=seed + experiment_num
                        )
                        target = metadata['target_node']
                    
                    # Compute PageRank with specified beta
                    pagerank = compute_pagerank(G_spam, alpha=beta, max_iter=100)
                    target_pr = pagerank.get(target, 0.0)
                    
                    # Calculate theoretical PageRank
                    theoretical_pr = calculate_theoretical_pagerank_boost(
                        m=m, n=G_spam.number_of_nodes(), beta=beta, x=0.01
                    )
                    
                    # Calculate boost factor (relative to average)
                    avg_pr = np.mean(list(pagerank.values()))
                    boost_factor = target_pr / avg_pr if avg_pr > 0 else 0.0
                    
                    # Calculate error
                    error = abs(target_pr - theoretical_pr)
                    relative_error = error / theoretical_pr if theoretical_pr > 0 else 0.0
                    
                    results.append({
                        'm': m,
                        'external_links': external,
                        'beta': beta,
                        'target_pagerank': target_pr,
                        'theoretical_pagerank': theoretical_pr,
                        'boost_factor': boost_factor,
                        'absolute_error': error,
                        'relative_error': relative_error,
                        'spam_type': spam_type
                    })
                
                except Exception as e:
                    logger.error(f"Experiment failed (m={m}, external={external}, beta={beta}): {e}")
                    continue
    
    df = pd.DataFrame(results)
    logger.info(f"Experiments complete: {len(df)} successful experiments")
    
    return df


def compare_spam_effectiveness(
    G: nx.Graph,
    spam_types: List[str] = ['simple', 'self_linking', 'dead_end', 'dual_link'],
    m: int = 1000,
    external: int = 10,
    seed: int = 42
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Compare effectiveness of different spam farm types.
    
    Create each spam type on fresh copy of G, compute PageRank,
    and compare target PageRank across types.
    
    Args:
        G: Original graph
        spam_types: List of spam types to compare
        m: Number of supporting pages
        external: Number of external links
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (comparison_DataFrame, pagerank_dict)
    """
    logger.info(f"Comparing spam effectiveness: {spam_types}")
    
    from spam_farm_generator import inject_spam_into_network
    from centrality_analysis import compute_pagerank
    
    results = []
    pagerank_dict = {}
    
    for spam_type in spam_types:
        try:
            logger.info(f"Testing spam type: {spam_type}")
            
            # Create spam farm
            if spam_type == 'self_linking':
                G_spam, metadata = create_self_linking_spam_farm(
                    G, m=m, external_links=external, seed=seed
                )
            elif spam_type == 'dead_end':
                G_spam, metadata = create_dead_end_spam_farm(
                    G, m=m, external_links=external, seed=seed
                )
            elif spam_type == 'dual_link':
                G_spam, metadata = create_dual_link_spam_farm(
                    G, m=m, external_links=external, seed=seed
                )
            elif spam_type == 'honeypot':
                G_spam, metadata = create_honeypot_spam_farm(
                    G, m=m, external_links=external, seed=seed
                )
            elif spam_type == 'layered':
                G_spam, metadata = create_layered_spam_farm(
                    G, layers=3, nodes_per_layer=m//3, external_links=external, seed=seed
                )
            else:
                # Use base spam farm generator
                G_spam, metadata = inject_spam_into_network(
                    G, spam_type=spam_type, m=m, external_links=external, seed=seed
                )
            
            target = metadata['target_node']
            
            # Compute PageRank
            pagerank = compute_pagerank(G_spam, alpha=0.85, max_iter=100)
            target_pr = pagerank.get(target, 0.0)
            
            # Calculate boost
            avg_pr = np.mean(list(pagerank.values()))
            boost_factor = target_pr / avg_pr if avg_pr > 0 else 0.0
            
            # Theoretical prediction
            theoretical_pr = calculate_theoretical_pagerank_boost(
                m=m, n=G_spam.number_of_nodes(), beta=0.85, x=0.01
            )
            
            results.append({
                'spam_type': spam_type,
                'target_pagerank': target_pr,
                'theoretical_pagerank': theoretical_pr,
                'boost_factor': boost_factor,
                'num_spam_nodes': len(metadata.get('spam_nodes', [])),
                'external_links': metadata.get('external_links', 0)
            })
            
            pagerank_dict[spam_type] = target_pr
            
            logger.info(f"  {spam_type}: PageRank={target_pr:.6f}, Boost={boost_factor:.2f}x")
        
        except Exception as e:
            logger.error(f"Failed to test {spam_type}: {e}")
            continue
    
    df = pd.DataFrame(results)
    df = df.sort_values('target_pagerank', ascending=False)
    
    logger.info(f"Comparison complete: {len(df)} spam types tested")
    
    return df, pagerank_dict


def plot_spam_comparison(
    comparison_df: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Plot comparison of different spam farm types.
    
    Args:
        comparison_df: DataFrame from compare_spam_effectiveness
        save_path: Optional path to save figure
        figsize: Figure size tuple
    """
    if comparison_df.empty:
        logger.warning("Comparison DataFrame is empty")
        return
    
    logger.info("Plotting spam comparison...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: PageRank comparison
    ax1.bar(comparison_df['spam_type'], comparison_df['target_pagerank'], color='steelblue')
    ax1.axhline(comparison_df['theoretical_pagerank'].iloc[0], color='red', 
               linestyle='--', label='Theoretical')
    ax1.set_xlabel('Spam Type', fontsize=12)
    ax1.set_ylabel('Target PageRank', fontsize=12)
    ax1.set_title('PageRank Comparison', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Boost factor comparison
    ax2.bar(comparison_df['spam_type'], comparison_df['boost_factor'], color='orange')
    ax2.set_xlabel('Spam Type', fontsize=12)
    ax2.set_ylabel('Boost Factor', fontsize=12)
    ax2.set_title('Boost Factor Comparison', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Comparison plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """
    Main function demonstrating spam farm variants.
    """
    print("=" * 60)
    print("Spam Farm Variants - Demo")
    print("=" * 60)
    
    try:
        import networkx as nx
        
        # Create sample graph
        print("\n1. Creating sample graph...")
        G = nx.erdos_renyi_graph(n=500, p=0.01, seed=42)
        print(f"   Created graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Test self-linking spam farm
        print("\n2. Testing self-linking spam farm...")
        G_self, metadata_self = create_self_linking_spam_farm(G, m=50, external_links=5, seed=42)
        print(f"   Created: {len(metadata_self['spam_nodes'])} spam nodes")
        
        # Test dead-end spam farm
        print("\n3. Testing dead-end spam farm...")
        G_dead, metadata_dead = create_dead_end_spam_farm(G, m=50, external_links=5, seed=42)
        print(f"   Created: {len(metadata_dead['spam_nodes'])} spam nodes")
        
        # Test dual-link spam farm
        print("\n4. Testing dual-link spam farm...")
        G_dual, metadata_dual = create_dual_link_spam_farm(G, m=50, external_links=5, seed=42)
        print(f"   Created: {len(metadata_dual['spam_nodes'])} spam nodes")
        
        # Compare effectiveness
        print("\n5. Comparing spam effectiveness...")
        comparison_df, pr_dict = compare_spam_effectiveness(
            G, spam_types=['simple', 'self_linking', 'dead_end', 'dual_link'],
            m=100, external=10, seed=42
        )
        print(comparison_df.to_string(index=False))
        
        # Plot comparison
        plot_spam_comparison(comparison_df, save_path='results/figures/spam_comparison.png')
        print("   ✅ Comparison plot saved")
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)
    
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"\n✗ Error: {e}")
        raise


if __name__ == "__main__":
    main()

