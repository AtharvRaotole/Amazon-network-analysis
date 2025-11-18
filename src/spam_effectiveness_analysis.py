"""
Spam farm effectiveness analysis module.

This module analyzes the effectiveness of different spam farm strategies,
including PageRank amplification, ROI calculations, parameter sensitivity,
and network impact assessment.

Analysis capabilities:
1. PageRank amplification (actual vs theoretical)
2. Cost-benefit analysis (ROI)
3. Parameter sensitivity testing
4. Spam type comparison
5. Convergence analysis
6. Network damage assessment
7. Comprehensive visualizations

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
from typing import Dict, List, Set, Tuple, Optional, Union
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict
from scipy import stats
from scipy.optimize import minimize_scalar
import networkx as nx

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


def calculate_pagerank_amplification(
    G_original: nx.Graph,
    G_with_spam: nx.Graph,
    target_nodes: List[Union[int, str]],
    spam_metadata: Dict,
    beta: float = 0.85
) -> pd.DataFrame:
    """
    Calculate PageRank amplification for spam farm targets.
    
    Compares actual PageRank boost with theoretical prediction from textbook.
    Theoretical formula: y = x/(1-β²) + c*m/n where c = β/(1+β)
    
    Args:
        G_original: Original graph without spam
        G_with_spam: Graph with injected spam farms
        target_nodes: List of target node IDs
        spam_metadata: Dictionary with spam farm information
        beta: Damping factor for PageRank (default: 0.85)
    
    Returns:
        DataFrame with columns: target, pr_original, pr_with_spam, amplification,
                                predicted_amplification, error
    """
    logger.info("Calculating PageRank amplification...")
    
    # Calculate PageRank on both graphs
    logger.info("Computing PageRank on original graph...")
    pr_original = nx.pagerank(G_original, alpha=beta, max_iter=100)
    
    logger.info("Computing PageRank on graph with spam...")
    pr_with_spam = nx.pagerank(G_with_spam, alpha=beta, max_iter=100)
    
    results = []
    
    for target in tqdm(target_nodes, desc="Analyzing targets"):
        if target not in pr_original:
            logger.warning(f"Target {target} not in original graph")
            continue
        
        pr_orig = pr_original.get(target, 0.0)
        pr_spam = pr_with_spam.get(target, 0.0)
        
        if pr_orig == 0:
            amplification = np.inf if pr_spam > 0 else 1.0
        else:
            amplification = pr_spam / pr_orig
        
        # Get spam farm parameters from metadata
        farm_info = spam_metadata.get(target, {})
        m = farm_info.get('m', 0)  # Number of supporting pages
        n = len(G_original)  # Total nodes in original graph
        x = pr_orig  # External PageRank contribution (approximate)
        
        # Theoretical prediction: y = x/(1-β²) + c*m/n
        # where c = β/(1+β)
        c = beta / (1 + beta)
        predicted_pr = x / (1 - beta**2) + c * m / n
        
        if pr_orig == 0:
            predicted_amplification = np.inf if predicted_pr > 0 else 1.0
        else:
            predicted_amplification = predicted_pr / pr_orig
        
        error = abs(amplification - predicted_amplification) / predicted_amplification if predicted_amplification > 0 else 0.0
        
        results.append({
            'target': target,
            'pr_original': pr_orig,
            'pr_with_spam': pr_spam,
            'amplification': amplification,
            'predicted_amplification': predicted_amplification,
            'error': error,
            'm': m,
            'n': n,
            'boost': pr_spam - pr_orig
        })
    
    df = pd.DataFrame(results)
    
    logger.info(f"Amplification analysis completed: {len(df)} targets")
    if len(df) > 0:
        logger.info(f"  Average amplification: {df['amplification'].mean():.4f}")
        logger.info(f"  Average error: {df['error'].mean():.4f}")
    
    return df


def analyze_spam_farm_roi(
    spam_metadata: Dict,
    pagerank_boost: Dict[Union[int, str], float],
    cost_per_page: float = 1.0
) -> pd.DataFrame:
    """
    Calculate Return on Investment (ROI) for spam farms.
    
    Economic analysis of spam farm effectiveness:
    - Cost = number of spam pages * cost_per_page
    - Benefit = PageRank increase
    - ROI = benefit / cost
    
    Args:
        spam_metadata: Dictionary with spam farm information
        pagerank_boost: Dictionary mapping target to PageRank boost
        cost_per_page: Cost to create/maintain one spam page (default: 1.0)
    
    Returns:
        DataFrame with columns: target, cost, benefit, roi, cost_per_boost_unit
    """
    logger.info("Analyzing spam farm ROI...")
    
    results = []
    
    for target, boost in pagerank_boost.items():
        farm_info = spam_metadata.get(target, {})
        m = farm_info.get('m', 0)
        spam_nodes = farm_info.get('spam_nodes', [])
        total_spam_pages = len(spam_nodes) + 1  # Include target
        
        cost = total_spam_pages * cost_per_page
        benefit = boost
        
        if cost > 0:
            roi = benefit / cost
            cost_per_boost = cost / benefit if benefit > 0 else np.inf
        else:
            roi = np.inf if benefit > 0 else 0.0
            cost_per_boost = 0.0
        
        results.append({
            'target': target,
            'spam_pages': total_spam_pages,
            'm': m,
            'cost': cost,
            'benefit': benefit,
            'roi': roi,
            'cost_per_boost_unit': cost_per_boost,
            'farm_type': farm_info.get('farm_type', 'unknown')
        })
    
    df = pd.DataFrame(results)
    df = df.sort_values('roi', ascending=False)
    
    logger.info(f"ROI analysis completed: {len(df)} farms")
    if len(df) > 0:
        logger.info(f"  Best ROI: {df.iloc[0]['roi']:.4f} ({df.iloc[0]['target']})")
        logger.info(f"  Average ROI: {df['roi'].mean():.4f}")
    
    return df


def test_parameter_sensitivity(
    G_original: nx.Graph,
    m_values: List[int] = [100, 500, 1000, 5000],
    x_values: List[float] = [0.001, 0.01, 0.1],
    beta_values: List[float] = [0.8, 0.85, 0.9],
    target_node: Optional[Union[int, str]] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Test sensitivity of spam effectiveness to different parameters.
    
    For each parameter combination:
    - Create spam farm
    - Calculate PageRank boost
    - Measure detection rate (if detection methods available)
    
    Identifies optimal parameters for spammers and which parameters
    make detection easier.
    
    Args:
        G_original: Original graph
        m_values: List of m (supporting pages) values to test
        x_values: List of external PageRank contribution values
        beta_values: List of damping factor values
        target_node: Optional target node (if None, creates new node)
    
    Returns:
        Tuple of (sensitivity_dataframe, heatmap_figures_dict)
    """
    logger.info("Testing parameter sensitivity...")
    
    # Import spam farm generator
    try:
        from spam_farm_generator import create_simple_spam_farm
    except ImportError:
        logger.error("Cannot import spam_farm_generator module")
        raise
    
    results = []
    
    # Use a sample node as target if not provided
    if target_node is None:
        target_node = list(G_original.nodes())[0]
    
    # Test all combinations
    total_combinations = len(m_values) * len(x_values) * len(beta_values)
    
    with tqdm(total=total_combinations, desc="Testing parameters") as pbar:
        for m in m_values:
            for x in x_values:
                for beta in beta_values:
                    try:
                        # Create spam farm
                        G_copy = G_original.copy()
                        G_spam, spam_nodes, target = create_simple_spam_farm(
                            G_copy, target_node, m=m, external_links=int(10 * x * 1000)
                        )
                        
                        # Calculate PageRank on both graphs
                        pr_original = nx.pagerank(G_original, alpha=beta, max_iter=100)
                        pr_spam = nx.pagerank(G_spam, alpha=beta, max_iter=100)
                        
                        pr_orig = pr_original.get(target, 0.0)
                        pr_new = pr_spam.get(target, 0.0)
                        
                        boost = pr_new - pr_orig
                        amplification = pr_new / pr_orig if pr_orig > 0 else np.inf
                        
                        # Theoretical prediction
                        n = len(G_original)
                        c = beta / (1 + beta)
                        predicted_pr = x / (1 - beta**2) + c * m / n
                        predicted_boost = predicted_pr - x
                        
                        error = abs(boost - predicted_boost) / abs(predicted_boost) if predicted_boost != 0 else 0.0
                        
                        results.append({
                            'm': m,
                            'x': x,
                            'beta': beta,
                            'boost': boost,
                            'amplification': amplification,
                            'predicted_boost': predicted_boost,
                            'error': error,
                            'cost': (m + 1) * 1.0,  # Assuming cost_per_page=1
                            'roi': boost / (m + 1) if m > 0 else 0.0
                        })
                    
                    except Exception as e:
                        logger.warning(f"Error testing m={m}, x={x}, beta={beta}: {e}")
                    
                    pbar.update(1)
    
    df = pd.DataFrame(results)
    
    # Create heatmaps
    heatmaps = {}
    
    if len(df) > 0:
        # Heatmap: Boost vs (m, beta) for fixed x
        if len(x_values) > 0:
            x_fixed = x_values[len(x_values) // 2]
            df_x = df[df['x'] == x_fixed]
            if len(df_x) > 0:
                pivot = df_x.pivot_table(values='boost', index='m', columns='beta', aggfunc='mean')
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(pivot, annot=True, fmt='.6f', cmap='YlOrRd', ax=ax)
                ax.set_title(f'PageRank Boost vs (m, β) for x={x_fixed}', fontsize=14, fontweight='bold')
                ax.set_xlabel('Damping Factor (β)', fontsize=12)
                ax.set_ylabel('Supporting Pages (m)', fontsize=12)
                plt.tight_layout()
                heatmaps['boost_vs_m_beta'] = fig
        
        # Heatmap: ROI vs (m, beta)
        if len(x_values) > 0:
            df_x = df[df['x'] == x_fixed]
            if len(df_x) > 0:
                pivot = df_x.pivot_table(values='roi', index='m', columns='beta', aggfunc='mean')
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(pivot, annot=True, fmt='.6f', cmap='Greens', ax=ax)
                ax.set_title(f'ROI vs (m, β) for x={x_fixed}', fontsize=14, fontweight='bold')
                ax.set_xlabel('Damping Factor (β)', fontsize=12)
                ax.set_ylabel('Supporting Pages (m)', fontsize=12)
                plt.tight_layout()
                heatmaps['roi_vs_m_beta'] = fig
    
    logger.info(f"Parameter sensitivity analysis completed: {len(df)} combinations tested")
    
    return df, heatmaps


def compare_spam_types_effectiveness(
    G_original: nx.Graph,
    spam_types: List[str] = ['simple', 'multiple', 'collaborative'],
    total_m: int = 1000,
    num_farms: int = 1
) -> pd.DataFrame:
    """
    Compare effectiveness of different spam farm types.
    
    Creates each spam type with same total resources (total m) and compares:
    - PageRank boost
    - Detection difficulty
    - Cost-effectiveness
    
    Args:
        G_original: Original graph
        spam_types: List of spam types to compare
        total_m: Total number of supporting pages to distribute
        num_farms: Number of farms to create (for multiple/collaborative)
    
    Returns:
        DataFrame with comparison results
    """
    logger.info(f"Comparing spam types: {spam_types}")
    
    try:
        from spam_farm_generator import (
            create_simple_spam_farm,
            create_multiple_spam_farms,
            create_collaborative_spam_farms
        )
    except ImportError:
        logger.error("Cannot import spam_farm_generator module")
        raise
    
    results = []
    
    for spam_type in spam_types:
        logger.info(f"Testing {spam_type} spam farm...")
        
        try:
            G_copy = G_original.copy()
            
            if spam_type == 'simple':
                # Single farm with all m pages
                target = list(G_original.nodes())[0]
                G_spam, spam_nodes, target_node = create_simple_spam_farm(
                    G_copy, target, m=total_m, external_links=10
                )
                targets = [target_node]
                m_per_farm = total_m
            
            elif spam_type == 'multiple':
                # Multiple independent farms
                m_per_farm = total_m // num_farms
                G_spam, farms_dict = create_multiple_spam_farms(
                    G_copy, num_farms=num_farms, m_per_farm=m_per_farm, external_links=5
                )
                targets = [farm_info[1] for farm_info in farms_dict.values()]  # Extract target nodes
            
            elif spam_type == 'collaborative':
                # Collaborative farms
                m_per_farm = total_m // num_farms
                G_spam, farms_dict, targets = create_collaborative_spam_farms(
                    G_copy, num_farms=num_farms, m_per_farm=m_per_farm, external_links=10
                )
            
            else:
                logger.warning(f"Unknown spam type: {spam_type}, skipping")
                continue
            
            # Calculate PageRank on both graphs
            pr_original = nx.pagerank(G_original, alpha=0.85, max_iter=100)
            pr_spam = nx.pagerank(G_spam, alpha=0.85, max_iter=100)
            
            # Calculate average boost per target
            boosts = []
            for target in targets:
                pr_orig = pr_original.get(target, 0.0)
                pr_new = pr_spam.get(target, 0.0)
                boosts.append(pr_new - pr_orig)
            
            avg_boost = np.mean(boosts) if boosts else 0.0
            total_boost = sum(boosts)
            
            # Calculate cost
            total_spam_pages = len(G_spam) - len(G_original)
            cost = total_spam_pages * 1.0  # cost_per_page = 1
            roi = total_boost / cost if cost > 0 else 0.0
            
            results.append({
                'spam_type': spam_type,
                'num_farms': num_farms if spam_type != 'simple' else 1,
                'm_per_farm': m_per_farm,
                'total_m': total_m,
                'total_spam_pages': total_spam_pages,
                'avg_boost_per_target': avg_boost,
                'total_boost': total_boost,
                'cost': cost,
                'roi': roi,
                'num_targets': len(targets)
            })
        
        except Exception as e:
            logger.error(f"Error testing {spam_type}: {e}", exc_info=True)
            continue
    
    df = pd.DataFrame(results)
    df = df.sort_values('roi', ascending=False)
    
    logger.info(f"Spam type comparison completed: {len(df)} types")
    if len(df) > 0:
        logger.info(f"  Best ROI: {df.iloc[0]['roi']:.6f} ({df.iloc[0]['spam_type']})")
    
    return df


def analyze_spam_convergence(
    G_with_spam: nx.Graph,
    target_nodes: List[Union[int, str]],
    max_iterations: int = 100,
    beta: float = 0.85,
    tol: float = 1e-6
) -> Tuple[pd.DataFrame, plt.Figure]:
    """
    Analyze PageRank convergence for spam farm targets.
    
    Tracks PageRank of targets over iterations to determine:
    - How many iterations until boost stabilizes
    - Whether spam causes oscillation
    - Convergence rate
    
    Args:
        G_with_spam: Graph with injected spam
        target_nodes: List of target node IDs to track
        max_iterations: Maximum iterations for PageRank
        beta: Damping factor
        tol: Convergence tolerance
    
    Returns:
        Tuple of (convergence_dataframe, convergence_plot_figure)
    """
    logger.info("Analyzing spam convergence...")
    
    # Custom PageRank with iteration tracking
    def pagerank_with_tracking(G, targets, alpha=0.85, max_iter=100, tol=1e-6):
        """PageRank that tracks target nodes over iterations."""
        n = len(G)
        if n == 0:
            return {}, {}
        
        # Initialize
        p = {node: 1.0 / n for node in G.nodes()}
        target_history = {target: [] for target in targets if target in G}
        
        for iteration in range(max_iter):
            p_new = {}
            for node in G.nodes():
                p_new[node] = (1 - alpha) / n
                for neighbor in G.neighbors(node):
                    p_new[node] += alpha * p[neighbor] / G.degree(neighbor)
            
            # Track target nodes
            for target in target_history.keys():
                target_history[target].append(p_new.get(target, 0.0))
            
            # Check convergence
            diff = sum(abs(p_new[node] - p[node]) for node in G.nodes())
            if diff < tol:
                logger.info(f"Converged after {iteration + 1} iterations")
                break
            
            p = p_new
        
        return p, target_history
    
    pr_final, target_history = pagerank_with_tracking(
        G_with_spam, target_nodes, alpha=beta, max_iter=max_iterations, tol=tol
    )
    
    # Create convergence DataFrame
    convergence_data = []
    for target, history in target_history.items():
        for iteration, pr_value in enumerate(history):
            convergence_data.append({
                'target': target,
                'iteration': iteration,
                'pagerank': pr_value
            })
    
    df = pd.DataFrame(convergence_data)
    
    # Create convergence plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for target in target_history.keys():
        target_df = df[df['target'] == target]
        ax.plot(target_df['iteration'], target_df['pagerank'], 
               marker='o', markersize=3, label=f'Target {target}', linewidth=2)
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('PageRank', fontsize=12)
    ax.set_title('PageRank Convergence for Spam Farm Targets', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    logger.info(f"Convergence analysis completed: {len(target_history)} targets tracked")
    
    return df, fig


def analyze_network_damage(
    G_original: nx.Graph,
    G_with_spam: nx.Graph
) -> Dict:
    """
    Analyze how spam affects overall network properties.
    
    Compares network metrics before and after spam injection:
    - Average clustering coefficient
    - Diameter
    - Degree distribution
    - Modularity of communities
    
    Args:
        G_original: Original graph
        G_with_spam: Graph with injected spam
    
    Returns:
        Dictionary with impact analysis results
    """
    logger.info("Analyzing network damage from spam...")
    
    results = {}
    
    # Average clustering coefficient
    logger.info("Computing clustering coefficients...")
    try:
        clustering_orig = nx.average_clustering(G_original)
        clustering_spam = nx.average_clustering(G_with_spam)
        results['clustering_original'] = clustering_orig
        results['clustering_with_spam'] = clustering_spam
        results['clustering_change'] = clustering_spam - clustering_orig
        results['clustering_change_pct'] = (clustering_spam - clustering_orig) / clustering_orig * 100 if clustering_orig > 0 else 0.0
    except Exception as e:
        logger.warning(f"Error computing clustering: {e}")
        results['clustering_original'] = None
        results['clustering_with_spam'] = None
    
    # Diameter (approximate for large graphs)
    logger.info("Computing diameter...")
    try:
        if nx.is_connected(G_original) and nx.is_connected(G_with_spam):
            # Sample nodes for diameter calculation
            sample_size = min(100, len(G_original))
            sample_nodes = list(G_original.nodes())[:sample_size]
            
            def approximate_diameter(G, sample_nodes):
                max_path = 0
                for i, node1 in enumerate(sample_nodes):
                    for node2 in sample_nodes[i+1:]:
                        try:
                            path_length = nx.shortest_path_length(G, node1, node2)
                            max_path = max(max_path, path_length)
                        except nx.NetworkXNoPath:
                            continue
                return max_path
            
            diameter_orig = approximate_diameter(G_original, sample_nodes)
            diameter_spam = approximate_diameter(G_with_spam, sample_nodes)
            results['diameter_original'] = diameter_orig
            results['diameter_with_spam'] = diameter_spam
            results['diameter_change'] = diameter_spam - diameter_orig
        else:
            results['diameter_original'] = None
            results['diameter_with_spam'] = None
    except Exception as e:
        logger.warning(f"Error computing diameter: {e}")
        results['diameter_original'] = None
        results['diameter_with_spam'] = None
    
    # Degree distribution
    logger.info("Analyzing degree distribution...")
    degrees_orig = [d for n, d in G_original.degree()]
    degrees_spam = [d for n, d in G_with_spam.degree()]
    
    results['avg_degree_original'] = np.mean(degrees_orig)
    results['avg_degree_with_spam'] = np.mean(degrees_spam)
    results['avg_degree_change'] = np.mean(degrees_spam) - np.mean(degrees_orig)
    
    # Statistical test for degree distribution change
    try:
        stat, p_value = stats.ks_2samp(degrees_orig, degrees_spam)
        results['degree_distribution_ks_statistic'] = stat
        results['degree_distribution_p_value'] = p_value
        results['degree_distribution_significant'] = p_value < 0.05
    except Exception as e:
        logger.warning(f"Error in KS test: {e}")
    
    # Modularity (if communities detected)
    logger.info("Computing modularity...")
    try:
        import community as community_louvain
        communities_orig = community_louvain.best_partition(G_original)
        communities_spam = community_louvain.best_partition(G_with_spam)
        
        modularity_orig = community_louvain.modularity(communities_orig, G_original)
        modularity_spam = community_louvain.modularity(communities_spam, G_with_spam)
        
        results['modularity_original'] = modularity_orig
        results['modularity_with_spam'] = modularity_spam
        results['modularity_change'] = modularity_spam - modularity_orig
    except Exception as e:
        logger.warning(f"Error computing modularity: {e}")
        results['modularity_original'] = None
        results['modularity_with_spam'] = None
    
    # Basic statistics
    results['nodes_original'] = len(G_original)
    results['nodes_with_spam'] = len(G_with_spam)
    results['edges_original'] = G_original.number_of_edges()
    results['edges_with_spam'] = G_with_spam.number_of_edges()
    results['spam_nodes_added'] = len(G_with_spam) - len(G_original)
    results['spam_edges_added'] = G_with_spam.number_of_edges() - G_original.number_of_edges()
    
    logger.info("Network damage analysis completed")
    logger.info(f"  Nodes added: {results['spam_nodes_added']}")
    logger.info(f"  Edges added: {results['spam_edges_added']}")
    
    return results


def visualize_spam_impact(
    G_original: nx.Graph,
    G_with_spam: nx.Graph,
    spam_metadata: Dict,
    output_dir: str
) -> Dict[str, str]:
    """
    Create comprehensive visualizations of spam impact.
    
    Generates before/after visualizations:
    - Top-100 ranking changes
    - PageRank distribution shift
    - Network structure changes
    - Spam farm locations in network
    
    Args:
        G_original: Original graph
        G_with_spam: Graph with injected spam
        spam_metadata: Dictionary with spam farm information
        output_dir: Directory to save plots
    
    Returns:
        Dictionary mapping plot_name to file_path
    """
    logger.info("Creating spam impact visualizations...")
    
    os.makedirs(output_dir, exist_ok=True)
    plot_paths = {}
    
    # 1. Top-100 ranking changes
    logger.info("Plotting ranking changes...")
    pr_original = nx.pagerank(G_original, alpha=0.85, max_iter=100)
    pr_spam = nx.pagerank(G_with_spam, alpha=0.85, max_iter=100)
    
    # Get top-100 from original
    top_100_orig = sorted(pr_original.items(), key=lambda x: x[1], reverse=True)[:100]
    top_100_orig_nodes = [node for node, _ in top_100_orig]
    
    # Get their ranks in spam graph
    all_nodes_spam = sorted(pr_spam.items(), key=lambda x: x[1], reverse=True)
    spam_ranks = {node: rank for rank, (node, _) in enumerate(all_nodes_spam, 1)}
    
    orig_ranks = list(range(1, 101))
    new_ranks = [spam_ranks.get(node, len(G_with_spam)) for node in top_100_orig_nodes]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(orig_ranks, new_ranks, alpha=0.6, s=50)
    ax.plot([1, 100], [1, 100], 'r--', linewidth=2, label='No change')
    ax.set_xlabel('Original Rank', fontsize=12)
    ax.set_ylabel('New Rank (with spam)', fontsize=12)
    ax.set_title('Top-100 Ranking Changes After Spam Injection', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    ranking_path = os.path.join(output_dir, 'ranking_changes.png')
    plt.savefig(ranking_path, dpi=300, bbox_inches='tight')
    plot_paths['ranking_changes'] = ranking_path
    plt.close()
    
    # 2. PageRank distribution shift
    logger.info("Plotting PageRank distribution shift...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    pr_values_orig = list(pr_original.values())
    pr_values_spam = list(pr_spam.values())
    
    ax1.hist(pr_values_orig, bins=50, alpha=0.7, label='Original', color='blue')
    ax1.hist(pr_values_spam, bins=50, alpha=0.7, label='With Spam', color='red')
    ax1.set_xlabel('PageRank', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('PageRank Distribution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.set_yscale('log')
    
    # Log-log plot
    ax2.scatter(pr_values_orig, pr_values_spam, alpha=0.3, s=10)
    ax2.plot([min(pr_values_orig), max(pr_values_orig)], 
            [min(pr_values_orig), max(pr_values_orig)], 'r--', linewidth=2)
    ax2.set_xlabel('Original PageRank', fontsize=12)
    ax2.set_ylabel('PageRank with Spam', fontsize=12)
    ax2.set_title('PageRank Before vs After', fontsize=12, fontweight='bold')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    distribution_path = os.path.join(output_dir, 'pagerank_distribution.png')
    plt.savefig(distribution_path, dpi=300, bbox_inches='tight')
    plot_paths['pagerank_distribution'] = distribution_path
    plt.close()
    
    # 3. Spam farm locations (if metadata available)
    if spam_metadata:
        logger.info("Plotting spam farm locations...")
        target_nodes = []
        spam_nodes = []
        
        for target, info in spam_metadata.items():
            target_nodes.append(target)
            spam_nodes.extend(info.get('spam_nodes', []))
        
        # Sample network for visualization
        sample_size = min(1000, len(G_with_spam))
        sample_nodes = list(G_with_spam.nodes())[:sample_size]
        G_sample = G_with_spam.subgraph(sample_nodes)
        
        # Create layout
        pos = nx.spring_layout(G_sample, k=1, iterations=50, seed=42)
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Draw network
        nx.draw_networkx_nodes(G_sample, pos, node_color='lightblue', 
                              node_size=20, alpha=0.6, ax=ax)
        nx.draw_networkx_edges(G_sample, pos, alpha=0.1, width=0.5, ax=ax)
        
        # Highlight spam nodes
        spam_in_sample = [n for n in spam_nodes if n in G_sample]
        if spam_in_sample:
            nx.draw_networkx_nodes(G_sample, pos, nodelist=spam_in_sample,
                                  node_color='red', node_size=100, alpha=0.8, ax=ax)
        
        # Highlight targets
        targets_in_sample = [n for n in target_nodes if n in G_sample]
        if targets_in_sample:
            nx.draw_networkx_nodes(G_sample, pos, nodelist=targets_in_sample,
                                  node_color='orange', node_size=200, alpha=0.9, ax=ax)
        
        ax.set_title('Spam Farm Locations in Network', fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        
        locations_path = os.path.join(output_dir, 'spam_locations.png')
        plt.savefig(locations_path, dpi=300, bbox_inches='tight')
        plot_paths['spam_locations'] = locations_path
        plt.close()
    
    logger.info(f"Visualizations saved to {output_dir}")
    
    return plot_paths


def main():
    """
    Main function demonstrating spam effectiveness analysis.
    """
    print("=" * 60)
    print("Spam Effectiveness Analysis Module - Demo")
    print("=" * 60)
    
    try:
        import networkx as nx
        
        # Create sample graph
        print("\n1. Creating sample graph...")
        G = nx.erdos_renyi_graph(1000, 0.01, seed=42)
        print(f"   Nodes: {len(G)}, Edges: {G.number_of_edges()}")
        
        # Simulate spam injection
        print("\n2. Simulating spam injection...")
        G_spam = G.copy()
        target = list(G.nodes())[0]
        spam_nodes = list(range(10000, 10100))  # 100 spam nodes
        
        # Add spam farm structure
        for spam_node in spam_nodes:
            G_spam.add_edge(target, spam_node)
            G_spam.add_edge(spam_node, target)
        
        spam_metadata = {
            target: {
                'm': len(spam_nodes),
                'spam_nodes': spam_nodes,
                'farm_type': 'simple'
            }
        }
        
        print(f"   Spam nodes added: {len(spam_nodes)}")
        
        # Test amplification
        print("\n3. Testing PageRank amplification...")
        amplification_df = calculate_pagerank_amplification(
            G, G_spam, [target], spam_metadata
        )
        print(amplification_df[['target', 'amplification', 'predicted_amplification']].to_string(index=False))
        
        # Test ROI
        print("\n4. Analyzing ROI...")
        pr_original = nx.pagerank(G, alpha=0.85)
        pr_spam = nx.pagerank(G_spam, alpha=0.85)
        boost = {target: pr_spam[target] - pr_original[target]}
        
        roi_df = analyze_spam_farm_roi(spam_metadata, boost)
        print(roi_df[['target', 'cost', 'benefit', 'roi']].to_string(index=False))
        
        # Test network damage
        print("\n5. Analyzing network damage...")
        damage = analyze_network_damage(G, G_spam)
        print(f"   Clustering change: {damage.get('clustering_change', 'N/A')}")
        print(f"   Average degree change: {damage.get('avg_degree_change', 'N/A'):.4f}")
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)
    
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"\n✗ Error: {e}")
        raise


if __name__ == "__main__":
    main()

