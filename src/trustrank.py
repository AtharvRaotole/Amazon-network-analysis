"""
TrustRank computation module for network analysis.

This module implements TrustRank algorithm based on textbook Section 5.4.4.
TrustRank is a topic-sensitive PageRank where teleport is restricted to
trusted pages, helping identify spam by comparing with regular PageRank.

Reference: Mining of Massive Datasets, Chapter 5.4.4
"""

import os
import time
import logging
import json
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Set, Tuple, Optional, Union
from datetime import datetime
from tqdm import tqdm

# Import centrality functions
from centrality_analysis import compute_pagerank, compute_degree_centrality

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


def select_trusted_pages(
    G: nx.Graph,
    method: str = 'top_pagerank',
    k: int = 100,
    pagerank_scores: Optional[Dict] = None,
    manual_nodes: Optional[List] = None
) -> Set[Union[int, str]]:
    """
    Select trusted pages using various methods.
    
    Methods:
    - 'top_pagerank': Select top-k by PageRank from original graph
    - 'high_degree': Select top-k by degree centrality
    - 'manual': Use provided list of trusted nodes
    
    Args:
        G: NetworkX graph
        method: Selection method ('top_pagerank', 'high_degree', 'manual')
        k: Number of trusted pages to select (for top_pagerank and high_degree)
        pagerank_scores: Optional pre-computed PageRank scores
        manual_nodes: Optional list of manually specified trusted nodes
    
    Returns:
        Set of trusted node IDs
    
    Raises:
        ValueError: If method is unknown or parameters are invalid
    """
    if method not in ['top_pagerank', 'high_degree', 'manual']:
        raise ValueError(f"Unknown method: {method}. Must be 'top_pagerank', 'high_degree', or 'manual'")
    
    logger.info(f"Selecting trusted pages: method={method}, k={k}")
    
    if method == 'manual':
        if manual_nodes is None:
            raise ValueError("manual_nodes must be provided for 'manual' method")
        trusted = set(manual_nodes)
        logger.info(f"Using {len(trusted)} manually specified trusted nodes")
    
    elif method == 'top_pagerank':
        # Compute PageRank if not provided
        if pagerank_scores is None:
            logger.info("Computing PageRank for trusted page selection...")
            pagerank_scores = compute_pagerank(G, alpha=0.85, max_iter=100)
        
        # Get top-k nodes
        sorted_nodes = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)
        trusted = {node for node, _ in sorted_nodes[:k]}
        logger.info(f"Selected top {len(trusted)} nodes by PageRank")
    
    elif method == 'high_degree':
        # Get top-k by degree
        degrees = dict(G.degree())
        sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
        trusted = {node for node, _ in sorted_nodes[:k]}
        logger.info(f"Selected top {len(trusted)} nodes by degree")
    
    # Validate trusted nodes exist in graph
    valid_trusted = {node for node in trusted if node in G}
    if len(valid_trusted) < len(trusted):
        logger.warning(f"Removed {len(trusted) - len(valid_trusted)} invalid trusted nodes")
    
    if len(valid_trusted) == 0:
        logger.warning("No valid trusted nodes found, using random sample")
        all_nodes = list(G.nodes())
        valid_trusted = set(np.random.choice(all_nodes, size=min(k, len(all_nodes)), replace=False))
    
    logger.info(f"Final trusted set: {len(valid_trusted)} nodes")
    
    return valid_trusted


def compute_trustrank(
    G: nx.Graph,
    trusted_nodes: Set[Union[int, str]],
    beta: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6
) -> Tuple[Dict[Union[int, str], float], Dict]:
    """
    Compute TrustRank scores using topic-sensitive PageRank.
    
    TrustRank formula (Section 5.4.4):
    v' = β*M*v + (1-β)*e_S/|S|
    
    where:
    - M is the transition matrix
    - β is the damping factor
    - e_S is the indicator vector (1 for trusted nodes, 0 elsewhere)
    - |S| is the number of trusted nodes
    
    This is equivalent to PageRank with teleport restricted to trusted set.
    
    Args:
        G: NetworkX graph
        trusted_nodes: Set of trusted node IDs
        beta: Damping factor (default: 0.85)
        max_iter: Maximum iterations (default: 100)
        tol: Convergence tolerance (default: 1e-6)
    
    Returns:
        Tuple of (trustrank_dict, convergence_info) where:
            - trustrank_dict: Dictionary mapping node to TrustRank score
            - convergence_info: Dictionary with iteration count, convergence status
    
    Raises:
        ValueError: If trusted_nodes is empty or beta is invalid
    """
    if not trusted_nodes:
        raise ValueError("trusted_nodes must not be empty")
    if not 0 < beta < 1:
        raise ValueError("beta must be between 0 and 1")
    
    start_time = time.time()
    logger.info(f"Computing TrustRank: {len(trusted_nodes)} trusted nodes, beta={beta}")
    
    # Validate trusted nodes exist in graph
    valid_trusted = {node for node in trusted_nodes if node in G}
    if len(valid_trusted) < len(trusted_nodes):
        logger.warning(f"Removed {len(trusted_nodes) - len(valid_trusted)} invalid trusted nodes")
    
    if len(valid_trusted) == 0:
        raise ValueError("No valid trusted nodes in graph")
    
    # Use NetworkX topic-sensitive PageRank (personalized PageRank)
    # Create personalized vector: 1/|S| for trusted nodes, 0 elsewhere
    personalization = {node: 1.0 / len(valid_trusted) for node in valid_trusted}
    
    # Compute TrustRank using personalized PageRank
    try:
        trustrank_scores = nx.pagerank(
            G,
            alpha=beta,
            personalization=personalization,
            max_iter=max_iter,
            tol=tol
        )
        
        # Check convergence (NetworkX doesn't return iteration count directly)
        # We'll estimate based on typical convergence
        convergence_info = {
            'converged': True,
            'iterations': max_iter,  # NetworkX doesn't expose actual iterations
            'tolerance': tol
        }
    
    except Exception as e:
        logger.error(f"TrustRank computation failed: {e}")
        raise
    
    computation_time = time.time() - start_time
    
    # Normalize scores (should already be normalized by NetworkX)
    total_score = sum(trustrank_scores.values())
    if abs(total_score - 1.0) > 0.01:
        logger.warning(f"TrustRank scores don't sum to 1.0 (sum={total_score:.6f}), normalizing...")
        trustrank_scores = {node: score / total_score for node, score in trustrank_scores.items()}
    
    convergence_info['computation_time'] = computation_time
    convergence_info['total_score'] = sum(trustrank_scores.values())
    
    logger.info(f"TrustRank computed in {computation_time:.4f} seconds")
    logger.info(f"  TrustRank sum: {sum(trustrank_scores.values()):.6f}")
    logger.info(f"  Max score: {max(trustrank_scores.values()):.6f}")
    logger.info(f"  Min score: {min(trustrank_scores.values()):.6f}")
    
    return trustrank_scores, convergence_info


def compute_trustrank_multiple_sets(
    G: nx.Graph,
    trust_sets_dict: Dict[str, Set[Union[int, str]]],
    beta: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6
) -> Dict[str, Dict[Union[int, str], float]]:
    """
    Compute TrustRank for multiple trust sets.
    
    Args:
        G: NetworkX graph
        trust_sets_dict: Dictionary mapping set_name to set of trusted nodes
        beta: Damping factor (default: 0.85)
        max_iter: Maximum iterations (default: 100)
        tol: Convergence tolerance (default: 1e-6)
    
    Returns:
        Dictionary mapping set_name to trustrank_scores dictionary
    
    Raises:
        ValueError: If trust_sets_dict is empty
    """
    if not trust_sets_dict:
        raise ValueError("trust_sets_dict must not be empty")
    
    logger.info(f"Computing TrustRank for {len(trust_sets_dict)} trust sets...")
    
    results = {}
    
    for set_name, trusted_nodes in tqdm(trust_sets_dict.items(), desc="Computing TrustRank"):
        try:
            trustrank_scores, _ = compute_trustrank(
                G, trusted_nodes, beta=beta, max_iter=max_iter, tol=tol
            )
            results[set_name] = trustrank_scores
            logger.info(f"  {set_name}: {len(trusted_nodes)} trusted nodes")
        except Exception as e:
            logger.error(f"Failed to compute TrustRank for {set_name}: {e}")
            continue
    
    logger.info(f"Computed TrustRank for {len(results)} trust sets")
    
    return results


def compare_pagerank_trustrank(
    pagerank: Dict[Union[int, str], float],
    trustrank: Dict[Union[int, str], float],
    node_list: Optional[List[Union[int, str]]] = None
) -> pd.DataFrame:
    """
    Create comparison DataFrame between PageRank and TrustRank.
    
    Args:
        pagerank: Dictionary of PageRank scores
        trustrank: Dictionary of TrustRank scores
        node_list: Optional list of nodes to include (default: all nodes in both)
    
    Returns:
        DataFrame with columns: node_id, pagerank, trustrank, difference, ratio
        Sorted by difference (descending)
    """
    logger.info("Comparing PageRank and TrustRank...")
    
    # Get common nodes
    if node_list is None:
        common_nodes = set(pagerank.keys()) & set(trustrank.keys())
    else:
        common_nodes = set(node_list) & set(pagerank.keys()) & set(trustrank.keys())
    
    if not common_nodes:
        logger.warning("No common nodes between PageRank and TrustRank")
        return pd.DataFrame()
    
    # Create comparison data
    comparison_data = []
    for node in common_nodes:
        pr = pagerank.get(node, 0.0)
        tr = trustrank.get(node, 0.0)
        diff = pr - tr
        ratio = pr / tr if tr > 0 else np.inf
        
        comparison_data.append({
            'node_id': node,
            'pagerank': pr,
            'trustrank': tr,
            'difference': diff,
            'ratio': ratio
        })
    
    df = pd.DataFrame(comparison_data)
    df = df.sort_values('difference', ascending=False)
    
    logger.info(f"Comparison created: {len(df)} nodes")
    logger.info(f"  Average difference: {df['difference'].mean():.6f}")
    logger.info(f"  Max difference: {df['difference'].max():.6f}")
    logger.info(f"  Nodes with PR > TR: {len(df[df['difference'] > 0])}")
    logger.info(f"  Nodes with TR > PR: {len(df[df['difference'] < 0])}")
    
    return df


def identify_trusted_domains(
    G: nx.Graph,
    node_attributes: Optional[Dict] = None,
    domains: List[str] = ['edu', 'gov', 'org'],
    categories: Optional[List[str]] = None
) -> Set[Union[int, str]]:
    """
    Identify trusted nodes based on domain or category attributes.
    
    For web networks: Select nodes with trusted domains (.edu, .gov, .org)
    For Amazon network: Use product categories as proxy (Books, Electronics)
    
    Args:
        G: NetworkX graph
        node_attributes: Optional dictionary of node attributes
        domains: List of trusted domains (for web networks)
        categories: List of trusted categories (for Amazon network)
    
    Returns:
        Set of trusted node IDs
    """
    logger.info("Identifying trusted nodes by domain/category...")
    
    trusted = set()
    
    if node_attributes is None:
        # Try to get attributes from graph
        if G.number_of_nodes() > 0:
            sample_node = list(G.nodes())[0]
            if 'category' in G.nodes[sample_node]:
                # Amazon network - use categories
                if categories is None:
                    categories = ['Books', 'Electronics', 'Software']
                
                for node in G.nodes():
                    node_cat = G.nodes[node].get('category', '')
                    if any(cat.lower() in str(node_cat).lower() for cat in categories):
                        trusted.add(node)
                
                logger.info(f"Selected {len(trusted)} nodes by category: {categories}")
            elif 'domain' in G.nodes[sample_node]:
                # Web network - use domains
                for node in G.nodes():
                    node_domain = G.nodes[node].get('domain', '')
                    if any(dom in str(node_domain).lower() for dom in domains):
                        trusted.add(node)
                
                logger.info(f"Selected {len(trusted)} nodes by domain: {domains}")
            else:
                logger.warning("No category or domain attributes found, using high-degree nodes")
                degrees = dict(G.degree())
                top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:100]
                trusted = {node for node, _ in top_nodes}
    else:
        # Use provided attributes
        for node, attrs in node_attributes.items():
            if node in G:
                if 'category' in attrs and categories:
                    if any(cat.lower() in str(attrs['category']).lower() for cat in categories):
                        trusted.add(node)
                elif 'domain' in attrs:
                    if any(dom in str(attrs['domain']).lower() for dom in domains):
                        trusted.add(node)
    
    if len(trusted) == 0:
        logger.warning("No trusted nodes found, using top PageRank nodes as fallback")
        pagerank = compute_pagerank(G, alpha=0.85, max_iter=100)
        top_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:100]
        trusted = {node for node, _ in top_nodes}
    
    logger.info(f"Identified {len(trusted)} trusted nodes")
    
    return trusted


def visualize_trustrank_comparison(
    pagerank: Dict[Union[int, str], float],
    trustrank: Dict[Union[int, str], float],
    top_k: int = 50,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10)
) -> None:
    """
    Visualize PageRank vs TrustRank comparison.
    
    Creates scatter plot showing:
    - Points above diagonal = suspicious (high PR, low TR)
    - Points below diagonal = trusted (low PR, high TR)
    - Color coding by difference magnitude
    
    Args:
        pagerank: Dictionary of PageRank scores
        trustrank: Dictionary of TrustRank scores
        top_k: Number of top nodes to highlight (default: 50)
        save_path: Optional path to save figure
        figsize: Figure size tuple (default: (12, 10))
    """
    logger.info("Visualizing PageRank vs TrustRank comparison...")
    
    # Get common nodes
    common_nodes = set(pagerank.keys()) & set(trustrank.keys())
    
    if not common_nodes:
        logger.warning("No common nodes for visualization")
        return
    
    # Prepare data
    pr_values = [pagerank[node] for node in common_nodes]
    tr_values = [trustrank[node] for node in common_nodes]
    differences = [pr - tr for pr, tr in zip(pr_values, tr_values)]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create scatter plot with color coding
    scatter = ax.scatter(
        pr_values, tr_values,
        c=differences,
        cmap='RdYlGn',
        alpha=0.6,
        s=50,
        edgecolors='black',
        linewidth=0.5
    )
    
    # Add diagonal line (PR = TR)
    min_val = min(min(pr_values), min(tr_values))
    max_val = max(max(pr_values), max(tr_values))
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='PR = TR')
    
    # Highlight top-k suspicious nodes (high PR, low TR)
    comparison_df = compare_pagerank_trustrank(pagerank, trustrank)
    if not comparison_df.empty and len(comparison_df) >= top_k:
        top_suspicious = comparison_df.head(top_k)
        for _, row in top_suspicious.iterrows():
            node = row['node_id']
            if node in common_nodes:
                ax.scatter(
                    pagerank[node], trustrank[node],
                    color='red', s=200, marker='*',
                    edgecolors='black', linewidth=1, zorder=5
                )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('PR - TR Difference', fontsize=12)
    
    # Formatting
    ax.set_xlabel('PageRank', fontsize=12)
    ax.set_ylabel('TrustRank', fontsize=12)
    ax.set_title('PageRank vs TrustRank Comparison\n(Points above diagonal = suspicious)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Add text annotations for interpretation
    ax.text(0.05, 0.95, 'Suspicious Region\n(High PR, Low TR)', 
           transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Comparison visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def save_trustrank_results(
    trustrank_scores: Dict[Union[int, str], float],
    trusted_nodes: Set[Union[int, str]],
    output_dir: str,
    pagerank_scores: Optional[Dict] = None
) -> None:
    """
    Save TrustRank results to files.
    
    Args:
        trustrank_scores: Dictionary of TrustRank scores
        trusted_nodes: Set of trusted node IDs
        output_dir: Directory to save results
        pagerank_scores: Optional PageRank scores for comparison
    """
    logger.info(f"Saving TrustRank results to {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save TrustRank scores as CSV
    trustrank_df = pd.DataFrame([
        {'node_id': node, 'trustrank': score}
        for node, score in trustrank_scores.items()
    ])
    trustrank_df = trustrank_df.sort_values('trustrank', ascending=False)
    trustrank_df.to_csv(os.path.join(output_dir, 'trustrank_scores.csv'), index=False)
    logger.info(f"TrustRank scores saved: {len(trustrank_df)} nodes")
    
    # Save trusted nodes as JSON
    trusted_list = sorted(list(trusted_nodes))
    with open(os.path.join(output_dir, 'trusted_nodes.json'), 'w') as f:
        json.dump({
            'trusted_nodes': [int(n) if isinstance(n, (int, np.integer)) else str(n) for n in trusted_list],
            'num_trusted': len(trusted_list),
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    logger.info(f"Trusted nodes saved: {len(trusted_list)} nodes")
    
    # Create statistics report
    stats_path = os.path.join(output_dir, 'trustrank_statistics.txt')
    with open(stats_path, 'w') as f:
        f.write("TrustRank Statistics Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write(f"Number of Trusted Nodes: {len(trusted_nodes):,}\n")
        f.write(f"Number of Nodes with TrustRank: {len(trustrank_scores):,}\n\n")
        f.write("TrustRank Score Statistics:\n")
        f.write(f"  Mean: {np.mean(list(trustrank_scores.values())):.6f}\n")
        f.write(f"  Median: {np.median(list(trustrank_scores.values())):.6f}\n")
        f.write(f"  Max: {max(trustrank_scores.values()):.6f}\n")
        f.write(f"  Min: {min(trustrank_scores.values()):.6f}\n")
        f.write(f"  Std: {np.std(list(trustrank_scores.values())):.6f}\n\n")
        
        # Top nodes
        f.write("Top 20 Nodes by TrustRank:\n")
        top_nodes = trustrank_df.head(20)
        for idx, row in top_nodes.iterrows():
            f.write(f"  {row['node_id']}: {row['trustrank']:.6f}\n")
        
        # Comparison with PageRank if available
        if pagerank_scores:
            f.write("\n" + "=" * 60 + "\n")
            f.write("Comparison with PageRank:\n\n")
            comparison_df = compare_pagerank_trustrank(pagerank_scores, trustrank_scores)
            if not comparison_df.empty:
                f.write("Top 10 Suspicious Nodes (High PR, Low TR):\n")
                top_suspicious = comparison_df.head(10)
                for idx, row in top_suspicious.iterrows():
                    f.write(f"  Node {row['node_id']}: PR={row['pagerank']:.6f}, "
                           f"TR={row['trustrank']:.6f}, Diff={row['difference']:.6f}\n")
    
    logger.info(f"Statistics report saved: {stats_path}")


def main():
    """
    Main function demonstrating TrustRank computation.
    """
    print("=" * 60)
    print("TrustRank Module - Demo")
    print("=" * 60)
    
    try:
        import networkx as nx
        
        # Create a sample graph
        print("\n1. Creating sample graph...")
        G = nx.erdos_renyi_graph(n=500, p=0.01, seed=42)
        print(f"   Created graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Select trusted pages
        print("\n2. Selecting trusted pages...")
        trusted = select_trusted_pages(G, method='top_pagerank', k=50)
        print(f"   Selected {len(trusted)} trusted nodes")
        
        # Compute TrustRank
        print("\n3. Computing TrustRank...")
        trustrank, conv_info = compute_trustrank(G, trusted, beta=0.85)
        print(f"   TrustRank computed: {len(trustrank)} nodes")
        print(f"   Computation time: {conv_info['computation_time']:.4f} seconds")
        
        # Compute PageRank for comparison
        print("\n4. Computing PageRank for comparison...")
        pagerank = compute_pagerank(G, alpha=0.85, max_iter=100)
        
        # Compare
        print("\n5. Comparing PageRank and TrustRank...")
        comparison_df = compare_pagerank_trustrank(pagerank, trustrank)
        print(comparison_df.head(10).to_string(index=False))
        
        # Visualize
        print("\n6. Visualizing comparison...")
        visualize_trustrank_comparison(
            pagerank, trustrank,
            top_k=20,
            save_path='results/figures/trustrank_comparison.png'
        )
        print("   ✅ Visualization saved")
        
        # Save results
        print("\n7. Saving results...")
        save_trustrank_results(
            trustrank, trusted, 'results/trustrank',
            pagerank_scores=pagerank
        )
        print("   ✅ Results saved")
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)
    
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"\n✗ Error: {e}")
        raise


if __name__ == "__main__":
    main()

