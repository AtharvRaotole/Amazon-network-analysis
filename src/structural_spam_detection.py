"""
Structural spam detection module for network analysis.

This module implements pattern-based spam detection methods that identify
suspicious network structures commonly associated with spam farms and
manipulation strategies.

Detection methods:
1. Reciprocal links: Tight clusters with high reciprocity
2. Star patterns: Central node with many supporters
3. Honeypot patterns: High in-degree, low out-degree
4. Clustering anomalies: Unusually high clustering coefficients
5. Isolated communities: Weakly connected communities
6. Link pattern analysis: Comprehensive structural metrics
7. Ensemble detection: Combined results from multiple methods

Reference: Mining of Massive Datasets, Chapter 5 (Link Analysis)
"""

import os
import time
import logging
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import Dict, List, Set, Tuple, Optional, Union
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict, Counter

# Import community detection
try:
    import community.community_louvain as community_louvain
except ImportError:
    try:
        import networkx.algorithms.community as nx_comm
        community_louvain = None
    except ImportError:
        community_louvain = None
        nx_comm = None

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


def detect_reciprocal_links(
    G: nx.Graph,
    min_cluster_size: int = 50,
    reciprocity_threshold: float = 0.8
) -> List[Dict]:
    """
    Detect tight clusters with high reciprocity.
    
    Spam farms often form tight clusters where most edges are bidirectional
    (high reciprocity). This method identifies such clusters.
    
    Args:
        G: NetworkX graph (directed or undirected)
        min_cluster_size: Minimum number of nodes in cluster (default: 50)
        reciprocity_threshold: Minimum reciprocity ratio (default: 0.8)
    
    Returns:
        List of dictionaries, each containing:
            - 'nodes': set of nodes in cluster
            - 'size': number of nodes
            - 'reciprocity': reciprocity ratio
            - 'internal_edges': number of internal edges
    """
    logger.info(f"Detecting reciprocal link clusters: min_size={min_cluster_size}, "
               f"reciprocity_threshold={reciprocity_threshold}")
    
    # Convert to directed if undirected (for reciprocity calculation)
    if not G.is_directed():
        G_dir = G.to_directed()
    else:
        G_dir = G
    
    suspicious_clusters = []
    
    # Use community detection to find clusters
    try:
        if community_louvain:
            communities = community_louvain.best_partition(G)
            # Convert to list of sets
            community_dict = defaultdict(set)
            for node, comm_id in communities.items():
                community_dict[comm_id].add(node)
            communities_list = list(community_dict.values())
        else:
            # Use NetworkX community detection
            communities_list = list(nx_comm.louvain_communities(G, seed=42))
    except Exception as e:
        logger.warning(f"Community detection failed, using connected components: {e}")
        communities_list = [set(comp) for comp in nx.connected_components(G)]
    
    logger.info(f"Found {len(communities_list)} communities")
    
    for i, cluster in enumerate(tqdm(communities_list, desc="Analyzing clusters")):
        if len(cluster) < min_cluster_size:
            continue
        
        # Calculate reciprocity for this cluster
        cluster_subgraph = G_dir.subgraph(cluster)
        total_edges = cluster_subgraph.number_of_edges()
        
        if total_edges == 0:
            continue
        
        # Count bidirectional edges
        bidirectional = 0
        for u, v in cluster_subgraph.edges():
            if cluster_subgraph.has_edge(v, u):
                bidirectional += 1
        
        # Reciprocity = bidirectional edges / total edges
        # For undirected graphs, all edges are bidirectional
        if not G.is_directed():
            reciprocity = 1.0
        else:
            reciprocity = bidirectional / total_edges if total_edges > 0 else 0.0
        
        if reciprocity >= reciprocity_threshold:
            suspicious_clusters.append({
                'cluster_id': i,
                'nodes': cluster,
                'size': len(cluster),
                'reciprocity': reciprocity,
                'internal_edges': total_edges,
                'density': nx.density(cluster_subgraph)
            })
    
    logger.info(f"Found {len(suspicious_clusters)} suspicious reciprocal clusters")
    for cluster in suspicious_clusters[:5]:  # Log top 5
        logger.info(f"  Cluster {cluster['cluster_id']}: {cluster['size']} nodes, "
                   f"reciprocity={cluster['reciprocity']:.3f}")
    
    return suspicious_clusters


def detect_star_patterns(
    G: nx.Graph,
    min_supporters: int = 100,
    max_out_degree_ratio: float = 10.0
) -> List[Tuple[Union[int, str], Set[Union[int, str]]]]:
    """
    Detect "star" patterns: one central node with many supporters.
    
    Spam farms often have a central target node with many supporting pages
    that primarily link to the center. This creates a star-like structure.
    
    Args:
        G: NetworkX graph (directed or undirected)
        min_supporters: Minimum number of supporters (default: 100)
        max_out_degree_ratio: Maximum out_degree/in_degree ratio for supporters (default: 10.0)
    
    Returns:
        List of tuples: (central_node, set_of_supporters)
    """
    logger.info(f"Detecting star patterns: min_supporters={min_supporters}, "
               f"max_out_degree_ratio={max_out_degree_ratio}")
    
    star_patterns = []
    
    # Convert to directed if needed
    if not G.is_directed():
        G_dir = G.to_directed()
    else:
        G_dir = G
    
    # For each node, check if it's a star center
    for center_node in tqdm(G_dir.nodes(), desc="Checking star patterns"):
        # Get neighbors
        neighbors = set(G_dir.neighbors(center_node))
        
        if len(neighbors) < min_supporters:
            continue
        
        # Check if neighbors primarily link to center
        supporters = set()
        
        for neighbor in neighbors:
            # For undirected graphs, check if neighbor has few other connections
            if not G.is_directed():
                neighbor_degree = G_dir.degree(neighbor)
                # If neighbor connects mostly to center's neighbors, it's a supporter
                center_neighbors = set(G_dir.neighbors(center_node))
                neighbor_neighbors = set(G_dir.neighbors(neighbor))
                overlap = len(center_neighbors & neighbor_neighbors)
                
                # Supporter if most connections are within center's neighborhood
                if neighbor_degree > 0:
                    overlap_ratio = overlap / neighbor_degree
                    if overlap_ratio > 0.7:  # 70% of connections within center's neighborhood
                        supporters.add(neighbor)
            else:
                # For directed graphs, check in/out degree ratio
                in_degree = G_dir.in_degree(neighbor)
                out_degree = G_dir.out_degree(neighbor)
                
                # Supporter if links primarily to center
                if in_degree > 0:
                    out_ratio = out_degree / in_degree
                    if out_ratio <= max_out_degree_ratio:
                        # Check if it links to center
                        if G_dir.has_edge(neighbor, center_node):
                            supporters.add(neighbor)
        
        if len(supporters) >= min_supporters:
            star_patterns.append((center_node, supporters))
            logger.debug(f"Star pattern: center={center_node}, supporters={len(supporters)}")
    
    logger.info(f"Found {len(star_patterns)} star patterns")
    if star_patterns:
        max_supporters = max(len(supporters) for _, supporters in star_patterns)
        logger.info(f"  Largest star: {max_supporters} supporters")
    
    return star_patterns


def detect_honeypot_patterns(
    G: nx.Graph,
    min_in_degree: int = 50,
    max_out_degree: int = 5
) -> List[Union[int, str]]:
    """
    Detect honeypot patterns: nodes with very high in-degree but low out-degree.
    
    Honeypot spam targets attract many links but link out to few pages,
    typical of spam farm targets.
    
    Args:
        G: NetworkX graph (directed or undirected)
        min_in_degree: Minimum in-degree threshold (default: 50)
        max_out_degree: Maximum out-degree threshold (default: 5)
    
    Returns:
        List of suspicious node IDs
    """
    logger.info(f"Detecting honeypot patterns: min_in_degree={min_in_degree}, "
               f"max_out_degree={max_out_degree}")
    
    suspicious_nodes = []
    
    # Convert to directed if needed
    if not G.is_directed():
        G_dir = G.to_directed()
    else:
        G_dir = G
    
    for node in tqdm(G_dir.nodes(), desc="Checking honeypot patterns"):
        in_degree = G_dir.in_degree(node)
        out_degree = G_dir.out_degree(node)
        
        if in_degree >= min_in_degree and out_degree <= max_out_degree:
            suspicious_nodes.append(node)
    
    logger.info(f"Found {len(suspicious_nodes)} honeypot pattern nodes")
    if suspicious_nodes:
        # Get statistics
        in_degrees = [G_dir.in_degree(n) for n in suspicious_nodes]
        out_degrees = [G_dir.out_degree(n) for n in suspicious_nodes]
        logger.info(f"  Mean in-degree: {np.mean(in_degrees):.1f}")
        logger.info(f"  Mean out-degree: {np.mean(out_degrees):.1f}")
    
    return suspicious_nodes


def calculate_clustering_anomalies(
    G: nx.Graph,
    z_score_threshold: float = 3.0
) -> List[Tuple[Union[int, str], float]]:
    """
    Detect nodes with unusually high clustering coefficients.
    
    Spam farms often have very high local clustering (neighbors are highly
    interconnected). This method identifies statistical outliers.
    
    Args:
        G: NetworkX graph
        z_score_threshold: Z-score threshold for anomaly detection (default: 3.0)
    
    Returns:
        List of tuples: (node_id, z_score)
    """
    logger.info(f"Calculating clustering anomalies: z_score_threshold={z_score_threshold}")
    
    # Calculate clustering coefficients
    clustering_coeffs = nx.clustering(G)
    
    if not clustering_coeffs:
        logger.warning("No clustering coefficients calculated")
        return []
    
    values = list(clustering_coeffs.values())
    mean_clustering = np.mean(values)
    std_clustering = np.std(values)
    
    if std_clustering == 0:
        logger.warning("Zero standard deviation in clustering coefficients")
        return []
    
    anomalies = []
    for node, clustering in clustering_coeffs.items():
        z_score = (clustering - mean_clustering) / std_clustering
        if z_score > z_score_threshold:
            anomalies.append((node, z_score))
    
    # Sort by z-score
    anomalies.sort(key=lambda x: x[1], reverse=True)
    
    logger.info(f"Found {len(anomalies)} clustering anomalies")
    if anomalies:
        logger.info(f"  Mean clustering: {mean_clustering:.4f}, Std: {std_clustering:.4f}")
        logger.info(f"  Top anomaly z-score: {anomalies[0][1]:.2f}")
    
    return anomalies


def detect_isolated_communities(
    G: nx.Graph,
    min_size: int = 50,
    max_external_links: int = 10,
    external_ratio_threshold: float = 0.1
) -> List[Dict]:
    """
    Detect isolated communities weakly connected to the rest of the graph.
    
    Spam farms often form isolated communities with few external connections.
    
    Args:
        G: NetworkX graph
        min_size: Minimum community size (default: 50)
        max_external_links: Maximum external links (default: 10)
        external_ratio_threshold: Maximum external link ratio (default: 0.1)
    
    Returns:
        List of dictionaries with community information
    """
    logger.info(f"Detecting isolated communities: min_size={min_size}, "
               f"max_external_links={max_external_links}")
    
    # Detect communities
    try:
        if community_louvain:
            communities = community_louvain.best_partition(G)
            community_dict = defaultdict(set)
            for node, comm_id in communities.items():
                community_dict[comm_id].add(node)
            communities_list = list(community_dict.values())
        else:
            communities_list = list(nx_comm.louvain_communities(G, seed=42))
    except Exception as e:
        logger.warning(f"Community detection failed: {e}")
        return []
    
    suspicious_communities = []
    
    for i, community in enumerate(tqdm(communities_list, desc="Analyzing communities")):
        if len(community) < min_size:
            continue
        
        # Count external links
        external_links = 0
        internal_links = 0
        
        for node in community:
            neighbors = set(G.neighbors(node))
            for neighbor in neighbors:
                if neighbor in community:
                    internal_links += 1
                else:
                    external_links += 1
        
        total_links = internal_links + external_links
        
        if total_links == 0:
            continue
        
        external_ratio = external_links / total_links
        
        # Check if isolated
        if external_links <= max_external_links and external_ratio <= external_ratio_threshold:
            suspicious_communities.append({
                'community_id': i,
                'nodes': community,
                'size': len(community),
                'internal_links': internal_links,
                'external_links': external_links,
                'external_ratio': external_ratio,
                'density': nx.density(G.subgraph(community))
            })
    
    logger.info(f"Found {len(suspicious_communities)} isolated communities")
    for comm in suspicious_communities[:5]:
        logger.info(f"  Community {comm['community_id']}: {comm['size']} nodes, "
                   f"external_ratio={comm['external_ratio']:.3f}")
    
    return suspicious_communities


def analyze_link_patterns(
    G: nx.Graph,
    node_subset: Optional[Set[Union[int, str]]] = None,
    sample_betweenness: bool = True,
    sample_size: int = 1000
) -> pd.DataFrame:
    """
    Analyze comprehensive link patterns for nodes.
    
    Calculates multiple structural metrics:
    - In-degree, Out-degree
    - In/Out ratio
    - Clustering coefficient
    - Betweenness centrality (sampled if large graph)
    - Local density
    
    Args:
        G: NetworkX graph
        node_subset: Optional subset of nodes to analyze (default: all nodes)
        sample_betweenness: Whether to sample for betweenness (default: True)
        sample_size: Sample size for betweenness if sampling (default: 1000)
    
    Returns:
        DataFrame with columns: node_id, in_degree, out_degree, in_out_ratio,
        clustering, betweenness, local_density
    """
    logger.info("Analyzing link patterns...")
    
    if node_subset is None:
        nodes_to_analyze = set(G.nodes())
    else:
        nodes_to_analyze = node_subset & set(G.nodes())
    
    logger.info(f"Analyzing {len(nodes_to_analyze)} nodes")
    
    # Convert to directed if needed
    if not G.is_directed():
        G_dir = G.to_directed()
    else:
        G_dir = G
    
    results = []
    
    # Calculate clustering (same for directed/undirected)
    clustering = nx.clustering(G)
    
    # Calculate betweenness (sample if large)
    if sample_betweenness and len(G) > sample_size:
        logger.info(f"Sampling {sample_size} nodes for betweenness centrality")
        sample_nodes = np.random.choice(list(G.nodes()), size=min(sample_size, len(G)), replace=False)
        betweenness = nx.betweenness_centrality(G, k=len(sample_nodes))
    else:
        betweenness = nx.betweenness_centrality(G)
    
    for node in tqdm(nodes_to_analyze, desc="Computing metrics"):
        in_degree = G_dir.in_degree(node)
        out_degree = G_dir.out_degree(node)
        
        # In/Out ratio
        in_out_ratio = in_degree / out_degree if out_degree > 0 else np.inf
        
        # Clustering
        node_clustering = clustering.get(node, 0.0)
        
        # Betweenness
        node_betweenness = betweenness.get(node, 0.0)
        
        # Local density (density of ego network)
        neighbors = set(G.neighbors(node))
        if len(neighbors) > 1:
            ego_graph = G.subgraph(neighbors | {node})
            local_density = nx.density(ego_graph)
        else:
            local_density = 0.0
        
        results.append({
            'node_id': node,
            'in_degree': in_degree,
            'out_degree': out_degree,
            'in_out_ratio': in_out_ratio,
            'clustering': node_clustering,
            'betweenness': node_betweenness,
            'local_density': local_density
        })
    
    df = pd.DataFrame(results)
    
    logger.info(f"Link pattern analysis completed: {len(df)} nodes")
    logger.info(f"  Mean clustering: {df['clustering'].mean():.4f}")
    logger.info(f"  Mean betweenness: {df['betweenness'].mean():.6f}")
    
    return df


def ensemble_structural_detection(
    G: nx.Graph,
    methods: List[str] = ['reciprocal', 'star', 'honeypot', 'clustering', 'isolated'],
    weights: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Run ensemble structural detection combining multiple methods.
    
    Nodes flagged by multiple methods are considered more suspicious.
    Results are weighted and ranked.
    
    Args:
        G: NetworkX graph
        methods: List of methods to use (default: all)
        weights: Optional dictionary of method weights (default: equal weights)
    
    Returns:
        DataFrame with columns: node_id, suspicion_score, methods_flagged, details
    """
    logger.info(f"Running ensemble structural detection: methods={methods}")
    
    if weights is None:
        weights = {method: 1.0 for method in methods}
    
    # Track which nodes are flagged by which methods
    node_flags = defaultdict(list)
    node_scores = defaultdict(float)
    
    # Run each detection method
    if 'reciprocal' in methods:
        logger.info("Running reciprocal link detection...")
        clusters = detect_reciprocal_links(G, min_cluster_size=50)
        for cluster in clusters:
            for node in cluster['nodes']:
                node_flags[node].append('reciprocal')
                node_scores[node] += weights.get('reciprocal', 1.0)
    
    if 'star' in methods:
        logger.info("Running star pattern detection...")
        stars = detect_star_patterns(G, min_supporters=100)
        for center, supporters in stars:
            node_flags[center].append('star_center')
            node_scores[center] += weights.get('star', 1.0)
            for supporter in supporters:
                node_flags[supporter].append('star_supporter')
                node_scores[supporter] += weights.get('star', 0.5)  # Lower weight for supporters
    
    if 'honeypot' in methods:
        logger.info("Running honeypot detection...")
        honeypots = detect_honeypot_patterns(G, min_in_degree=50, max_out_degree=5)
        for node in honeypots:
            node_flags[node].append('honeypot')
            node_scores[node] += weights.get('honeypot', 1.0)
    
    if 'clustering' in methods:
        logger.info("Running clustering anomaly detection...")
        anomalies = calculate_clustering_anomalies(G, z_score_threshold=3.0)
        for node, z_score in anomalies:
            node_flags[node].append('clustering_anomaly')
            # Weight by z-score
            node_scores[node] += weights.get('clustering', 1.0) * min(z_score / 3.0, 2.0)
    
    if 'isolated' in methods:
        logger.info("Running isolated community detection...")
        isolated = detect_isolated_communities(G, min_size=50)
        for comm in isolated:
            for node in comm['nodes']:
                node_flags[node].append('isolated_community')
                node_scores[node] += weights.get('isolated', 1.0)
    
    # Create results DataFrame
    results = []
    for node, score in node_scores.items():
        results.append({
            'node_id': node,
            'suspicion_score': score,
            'methods_flagged': len(node_flags[node]),
            'methods': ','.join(node_flags[node])
        })
    
    if not results:
        logger.warning("No suspicious nodes detected by any method")
        df = pd.DataFrame(columns=['node_id', 'suspicion_score', 'methods_flagged', 'methods'])
    else:
        df = pd.DataFrame(results)
        df = df.sort_values('suspicion_score', ascending=False)
    
    logger.info(f"Ensemble detection completed: {len(df)} suspicious nodes")
    logger.info(f"  Nodes flagged by multiple methods: {len(df[df['methods_flagged'] > 1])}")
    logger.info(f"  Top suspicion score: {df['suspicion_score'].max():.2f}")
    
    return df


def visualize_detected_structures(
    G: nx.Graph,
    detected_spam_dict: Dict[str, Union[List, Set, pd.DataFrame]],
    save_dir: str,
    max_nodes_per_plot: int = 500
) -> None:
    """
    Visualize detected spam structures for each detection method.
    
    Creates separate visualizations for each method and a combined overview.
    
    Args:
        G: NetworkX graph
        detected_spam_dict: Dictionary mapping method_name to detected nodes/structures
        save_dir: Directory to save visualizations
        max_nodes_per_plot: Maximum nodes to include in each plot (default: 500)
    """
    logger.info(f"Visualizing detected structures: {len(detected_spam_dict)} methods")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Create individual visualizations
    for method_name, detected in detected_spam_dict.items():
        logger.info(f"Visualizing {method_name}...")
        
        try:
            # Extract nodes based on detection type
            if isinstance(detected, (list, set)):
                nodes = set(detected)
            elif isinstance(detected, pd.DataFrame):
                if 'node_id' in detected.columns:
                    nodes = set(detected['node_id'].head(max_nodes_per_plot))
                elif 'nodes' in detected.columns:
                    nodes = set()
                    for node_set in detected['nodes'].head(max_nodes_per_plot):
                        nodes.update(node_set)
                else:
                    logger.warning(f"Unknown DataFrame format for {method_name}")
                    continue
            elif isinstance(detected, list) and len(detected) > 0:
                # Handle list of tuples (e.g., star patterns)
                if isinstance(detected[0], tuple):
                    nodes = set()
                    for item in detected[:10]:  # Top 10
                        if isinstance(item[1], set):
                            nodes.update(item[1])
                            nodes.add(item[0])
                        else:
                            nodes.add(item[0])
                else:
                    nodes = set(detected[:max_nodes_per_plot])
            else:
                logger.warning(f"Unknown format for {method_name}")
                continue
            
            if not nodes:
                logger.warning(f"No nodes to visualize for {method_name}")
                continue
            
            # Limit nodes for visualization
            if len(nodes) > max_nodes_per_plot:
                nodes = set(list(nodes)[:max_nodes_per_plot])
            
            # Create subgraph
            subgraph_nodes = nodes | set()
            # Add neighbors for context
            for node in list(nodes)[:100]:  # Limit expansion
                subgraph_nodes.update(list(G.neighbors(node))[:5])
            
            if len(subgraph_nodes) > max_nodes_per_plot:
                subgraph_nodes = set(list(subgraph_nodes)[:max_nodes_per_plot])
            
            subgraph = G.subgraph(subgraph_nodes)
            
            if subgraph.number_of_nodes() == 0:
                logger.warning(f"Empty subgraph for {method_name}")
                continue
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(14, 10))
            
            # Layout
            pos = nx.spring_layout(subgraph, k=1, iterations=50, seed=42)
            
            # Draw all nodes
            nx.draw_networkx_nodes(subgraph, pos, node_color='lightgray',
                                 node_size=50, alpha=0.5, ax=ax)
            
            # Highlight detected nodes
            detected_in_subgraph = nodes & set(subgraph.nodes())
            if detected_in_subgraph:
                nx.draw_networkx_nodes(subgraph, pos, nodelist=list(detected_in_subgraph),
                                     node_color='red', node_size=200, alpha=0.8, ax=ax)
            
            # Draw edges
            nx.draw_networkx_edges(subgraph, pos, alpha=0.2, width=0.5, ax=ax)
            
            # Labels for detected nodes only
            if len(detected_in_subgraph) <= 20:
                labels = {node: str(node) for node in detected_in_subgraph}
                nx.draw_networkx_labels(subgraph, pos, labels, font_size=8, ax=ax)
            
            ax.set_title(f'{method_name.replace("_", " ").title()} Detection\n'
                        f'{len(detected_in_subgraph)} detected nodes shown',
                        fontsize=14, fontweight='bold')
            ax.axis('off')
            
            plt.tight_layout()
            save_path = os.path.join(save_dir, f'{method_name}_detection.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"  Saved: {save_path}")
        
        except Exception as e:
            logger.error(f"Failed to visualize {method_name}: {e}", exc_info=True)
    
    # Create combined visualization
    logger.info("Creating combined visualization...")
    try:
        # Get all detected nodes
        all_detected = set()
        for detected in detected_spam_dict.values():
            if isinstance(detected, (list, set)):
                all_detected.update(detected)
            elif isinstance(detected, pd.DataFrame):
                if 'node_id' in detected.columns:
                    all_detected.update(detected['node_id'].head(100))
        
        if all_detected:
            # Sample for visualization
            if len(all_detected) > max_nodes_per_plot:
                all_detected = set(list(all_detected)[:max_nodes_per_plot])
            
            # Expand with neighbors
            subgraph_nodes = all_detected.copy()
            for node in list(all_detected)[:50]:
                subgraph_nodes.update(list(G.neighbors(node))[:3])
            
            if len(subgraph_nodes) > max_nodes_per_plot:
                subgraph_nodes = set(list(subgraph_nodes)[:max_nodes_per_plot])
            
            subgraph = G.subgraph(subgraph_nodes)
            
            if subgraph.number_of_nodes() > 0:
                fig, ax = plt.subplots(figsize=(16, 12))
                pos = nx.spring_layout(subgraph, k=1, iterations=50, seed=42)
                
                nx.draw_networkx_nodes(subgraph, pos, node_color='lightgray',
                                     node_size=30, alpha=0.3, ax=ax)
                
                detected_in_subgraph = all_detected & set(subgraph.nodes())
                if detected_in_subgraph:
                    nx.draw_networkx_nodes(subgraph, pos, nodelist=list(detected_in_subgraph),
                                         node_color='red', node_size=150, alpha=0.8, ax=ax)
                
                nx.draw_networkx_edges(subgraph, pos, alpha=0.1, width=0.3, ax=ax)
                
                ax.set_title(f'Combined Structural Spam Detection\n'
                            f'{len(detected_in_subgraph)} detected nodes across all methods',
                            fontsize=16, fontweight='bold')
                ax.axis('off')
                
                plt.tight_layout()
                save_path = os.path.join(save_dir, 'combined_detection.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"  Saved combined visualization: {save_path}")
    
    except Exception as e:
        logger.error(f"Failed to create combined visualization: {e}", exc_info=True)
    
    logger.info("Visualization completed")


def main():
    """
    Main function demonstrating structural spam detection.
    """
    print("=" * 60)
    print("Structural Spam Detection Module - Demo")
    print("=" * 60)
    
    try:
        import networkx as nx
        
        # Create a sample graph with spam-like structures
        print("\n1. Creating sample graph with spam patterns...")
        G = nx.erdos_renyi_graph(n=500, p=0.01, seed=42)
        
        # Add a star pattern
        center = max(G.nodes()) + 1
        supporters = list(range(center + 1, center + 101))
        G.add_node(center)
        for supp in supporters:
            G.add_node(supp)
            G.add_edge(center, supp)
            G.add_edge(supp, center)  # Reciprocal
        
        print(f"   Created graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Run detection methods
        print("\n2. Running detection methods...")
        
        detected = {}
        
        # Reciprocal links
        print("   - Reciprocal links...")
        clusters = detect_reciprocal_links(G, min_cluster_size=10)
        detected['reciprocal'] = clusters
        print(f"     Found {len(clusters)} clusters")
        
        # Star patterns
        print("   - Star patterns...")
        stars = detect_star_patterns(G, min_supporters=50)
        detected['star'] = stars
        print(f"     Found {len(stars)} star patterns")
        
        # Honeypot
        print("   - Honeypot patterns...")
        honeypots = detect_honeypot_patterns(G, min_in_degree=20, max_out_degree=5)
        detected['honeypot'] = honeypots
        print(f"     Found {len(honeypots)} honeypot nodes")
        
        # Clustering anomalies
        print("   - Clustering anomalies...")
        anomalies = calculate_clustering_anomalies(G, z_score_threshold=2.0)
        detected['clustering'] = [node for node, _ in anomalies[:50]]
        print(f"     Found {len(anomalies)} anomalies")
        
        # Isolated communities
        print("   - Isolated communities...")
        isolated = detect_isolated_communities(G, min_size=10)
        detected['isolated'] = isolated
        print(f"     Found {len(isolated)} isolated communities")
        
        # Ensemble
        print("\n3. Running ensemble detection...")
        ensemble_results = ensemble_structural_detection(G, methods=['reciprocal', 'star', 'honeypot'])
        print(f"   Found {len(ensemble_results)} suspicious nodes")
        print(ensemble_results.head(10).to_string(index=False))
        
        # Visualize
        print("\n4. Creating visualizations...")
        visualize_detected_structures(G, detected, 'results/figures/structural_detection')
        print("   ✅ Visualizations saved")
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)
    
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"\n✗ Error: {e}")
        raise


if __name__ == "__main__":
    main()

