"""
Spam Mass calculation module for network analysis.

This module implements Spam Mass calculation based on textbook Section 5.4.5.
Spam Mass measures how much of a page's PageRank comes from spam sources
by comparing PageRank with TrustRank.

Spam Mass formula: spam_mass = (PR - TR) / PR

Reference: Mining of Massive Datasets, Chapter 5.4.5
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
from scipy import stats

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


def calculate_spam_mass(
    pagerank_dict: Dict[Union[int, str], float],
    trustrank_dict: Dict[Union[int, str], float]
) -> Dict[Union[int, str], float]:
    """
    Calculate Spam Mass for each node.
    
    Spam Mass formula (Section 5.4.5):
    spam_mass = (PR - TR) / PR
    
    where:
    - PR is PageRank score
    - TR is TrustRank score
    
    Interpretation:
    - spam_mass = 0: All PageRank from trusted sources (legitimate)
    - spam_mass = 1: All PageRank from untrusted sources (spam)
    - spam_mass < 0: Higher TrustRank than PageRank (very trusted)
    
    Args:
        pagerank_dict: Dictionary mapping node to PageRank score
        trustrank_dict: Dictionary mapping node to TrustRank score
    
    Returns:
        Dictionary mapping node_id to spam_mass value
    
    Notes:
        - If PR = 0, spam_mass = 0 (no PageRank to analyze)
        - If spam_mass < 0, set to 0 (trusted pages)
    """
    logger.info("Calculating Spam Mass...")
    
    # Get common nodes
    common_nodes = set(pagerank_dict.keys()) & set(trustrank_dict.keys())
    
    if not common_nodes:
        logger.warning("No common nodes between PageRank and TrustRank")
        return {}
    
    spam_mass_dict = {}
    zero_pr_count = 0
    negative_sm_count = 0
    
    for node in common_nodes:
        pr = pagerank_dict.get(node, 0.0)
        tr = trustrank_dict.get(node, 0.0)
        
        if pr == 0:
            # No PageRank, cannot calculate spam mass
            spam_mass = 0.0
            zero_pr_count += 1
        else:
            spam_mass = (pr - tr) / pr
            
            # Clamp negative values to 0 (trusted pages)
            if spam_mass < 0:
                spam_mass = 0.0
                negative_sm_count += 1
        
        spam_mass_dict[node] = spam_mass
    
    logger.info(f"Spam Mass calculated for {len(spam_mass_dict)} nodes")
    logger.info(f"  Nodes with PR=0: {zero_pr_count}")
    logger.info(f"  Nodes with negative SM (clamped to 0): {negative_sm_count}")
    logger.info(f"  Mean spam mass: {np.mean(list(spam_mass_dict.values())):.4f}")
    logger.info(f"  Max spam mass: {max(spam_mass_dict.values()):.4f}")
    logger.info(f"  Nodes with SM > 0.7: {sum(1 for sm in spam_mass_dict.values() if sm > 0.7)}")
    
    return spam_mass_dict


def classify_by_spam_mass(
    spam_mass_dict: Dict[Union[int, str], float],
    threshold: float = 0.7,
    suspicious_threshold: float = 0.3
) -> Dict[Union[int, str], str]:
    """
    Classify nodes based on spam mass values.
    
    Classification:
    - 'likely_spam': spam_mass > threshold (default: 0.7)
    - 'suspicious': suspicious_threshold < spam_mass <= threshold (default: 0.3 < SM <= 0.7)
    - 'legitimate': spam_mass <= suspicious_threshold (default: <= 0.3)
    
    Args:
        spam_mass_dict: Dictionary mapping node to spam mass
        threshold: Threshold for likely spam classification (default: 0.7)
        suspicious_threshold: Threshold for suspicious classification (default: 0.3)
    
    Returns:
        Dictionary mapping node_id to classification string
    
    Raises:
        ValueError: If thresholds are invalid
    """
    if not 0 < suspicious_threshold < threshold <= 1:
        raise ValueError(f"Invalid thresholds: suspicious_threshold ({suspicious_threshold}) "
                        f"must be < threshold ({threshold}) <= 1")
    
    logger.info(f"Classifying nodes: threshold={threshold}, suspicious_threshold={suspicious_threshold}")
    
    classifications = {}
    
    for node, spam_mass in spam_mass_dict.items():
        if spam_mass > threshold:
            classifications[node] = 'likely_spam'
        elif spam_mass > suspicious_threshold:
            classifications[node] = 'suspicious'
        else:
            classifications[node] = 'legitimate'
    
    # Count classifications
    counts = pd.Series(classifications.values()).value_counts()
    logger.info(f"Classification results:")
    for cls, count in counts.items():
        logger.info(f"  {cls}: {count} nodes ({100*count/len(classifications):.1f}%)")
    
    return classifications


def optimize_spam_mass_threshold(
    spam_mass_dict: Dict[Union[int, str], float],
    true_spam_nodes: Set[Union[int, str]],
    true_legitimate_nodes: Set[Union[int, str]],
    threshold_range: Tuple[float, float] = (0.1, 0.9),
    step: float = 0.05
) -> Tuple[float, pd.DataFrame]:
    """
    Find optimal spam mass threshold to maximize F1-score.
    
    Tests multiple thresholds and calculates precision, recall, and F1-score
    for each threshold value.
    
    Args:
        spam_mass_dict: Dictionary mapping node to spam mass
        true_spam_nodes: Set of known spam node IDs
        true_legitimate_nodes: Set of known legitimate node IDs
        threshold_range: Tuple of (min, max) threshold values (default: (0.1, 0.9))
        step: Step size for threshold testing (default: 0.05)
    
    Returns:
        Tuple of (optimal_threshold, metrics_dataframe)
        metrics_dataframe contains columns: threshold, precision, recall, f1_score
    """
    logger.info("Optimizing spam mass threshold...")
    
    # Get all nodes with known labels
    all_labeled = true_spam_nodes | true_legitimate_nodes
    common_nodes = set(spam_mass_dict.keys()) & all_labeled
    
    if not common_nodes:
        logger.warning("No common nodes between spam_mass and true labels")
        return 0.7, pd.DataFrame()
    
    logger.info(f"Evaluating on {len(common_nodes)} labeled nodes")
    logger.info(f"  True spam: {len(true_spam_nodes & common_nodes)}")
    logger.info(f"  True legitimate: {len(true_legitimate_nodes & common_nodes)}")
    
    # Test different thresholds
    thresholds = np.arange(threshold_range[0], threshold_range[1] + step, step)
    results = []
    
    for threshold in tqdm(thresholds, desc="Testing thresholds"):
        # Classify nodes
        predicted_spam = {
            node for node, sm in spam_mass_dict.items()
            if node in common_nodes and sm > threshold
        }
        
        # Calculate metrics
        true_positives = len(predicted_spam & true_spam_nodes)
        false_positives = len(predicted_spam & true_legitimate_nodes)
        false_negatives = len((true_spam_nodes & common_nodes) - predicted_spam)
        true_negatives = len((true_legitimate_nodes & common_nodes) - predicted_spam)
        
        # Precision
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        
        # Recall
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        
        # F1-score
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Accuracy
        accuracy = (true_positives + true_negatives) / len(common_nodes) if len(common_nodes) > 0 else 0.0
        
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'true_negatives': true_negatives
        })
    
    metrics_df = pd.DataFrame(results)
    
    # Find optimal threshold (max F1-score)
    optimal_idx = metrics_df['f1_score'].idxmax()
    optimal_threshold = metrics_df.loc[optimal_idx, 'threshold']
    optimal_f1 = metrics_df.loc[optimal_idx, 'f1_score']
    
    logger.info(f"Optimal threshold: {optimal_threshold:.3f} (F1={optimal_f1:.4f})")
    logger.info(f"  Precision: {metrics_df.loc[optimal_idx, 'precision']:.4f}")
    logger.info(f"  Recall: {metrics_df.loc[optimal_idx, 'recall']:.4f}")
    
    return optimal_threshold, metrics_df


def analyze_spam_mass_distribution(
    spam_mass_dict: Dict[Union[int, str], float],
    true_spam_nodes: Optional[Set[Union[int, str]]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8)
) -> Tuple[Dict, plt.Figure]:
    """
    Analyze and visualize spam mass distribution.
    
    Creates histogram showing:
    - Overall spam mass distribution
    - Separate distributions for spam vs legitimate (if known)
    - Marked threshold values (0.5, 0.7, 0.9)
    - Statistical summary
    
    Args:
        spam_mass_dict: Dictionary mapping node to spam mass
        true_spam_nodes: Optional set of known spam node IDs
        save_path: Optional path to save figure
        figsize: Figure size tuple (default: (14, 8))
    
    Returns:
        Tuple of (statistics_dict, figure)
    """
    logger.info("Analyzing spam mass distribution...")
    
    spam_mass_values = list(spam_mass_dict.values())
    
    # Calculate statistics
    stats_dict = {
        'mean': np.mean(spam_mass_values),
        'median': np.median(spam_mass_values),
        'std': np.std(spam_mass_values),
        'min': min(spam_mass_values),
        'max': max(spam_mass_values),
        'percentile_25': np.percentile(spam_mass_values, 25),
        'percentile_75': np.percentile(spam_mass_values, 75),
        'percentile_90': np.percentile(spam_mass_values, 90),
        'percentile_95': np.percentile(spam_mass_values, 95),
        'percentile_99': np.percentile(spam_mass_values, 99)
    }
    
    logger.info(f"Spam Mass Statistics:")
    logger.info(f"  Mean: {stats_dict['mean']:.4f}")
    logger.info(f"  Median: {stats_dict['median']:.4f}")
    logger.info(f"  Std: {stats_dict['std']:.4f}")
    logger.info(f"  95th percentile: {stats_dict['percentile_95']:.4f}")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Overall distribution
    ax1 = axes[0]
    ax1.hist(spam_mass_values, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(0.5, color='orange', linestyle='--', linewidth=2, label='Threshold 0.5')
    ax1.axvline(0.7, color='red', linestyle='--', linewidth=2, label='Threshold 0.7')
    ax1.axvline(0.9, color='darkred', linestyle='--', linewidth=2, label='Threshold 0.9')
    ax1.set_xlabel('Spam Mass', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Spam Mass Distribution (All Nodes)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Separate distributions if true spam known
    ax2 = axes[1]
    if true_spam_nodes:
        spam_sm = [spam_mass_dict[node] for node in true_spam_nodes if node in spam_mass_dict]
        legitimate_sm = [spam_mass_dict[node] for node in spam_mass_dict.keys() 
                        if node not in true_spam_nodes]
        
        ax2.hist(legitimate_sm, bins=50, alpha=0.6, label='Legitimate', color='green', edgecolor='black')
        ax2.hist(spam_sm, bins=50, alpha=0.6, label='Spam', color='red', edgecolor='black')
        ax2.axvline(0.7, color='black', linestyle='--', linewidth=2, label='Threshold 0.7')
        ax2.set_xlabel('Spam Mass', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Spam Mass: Spam vs Legitimate', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # Statistical test
        if len(spam_sm) > 0 and len(legitimate_sm) > 0:
            try:
                t_stat, p_value = stats.mannwhitneyu(spam_sm, legitimate_sm, alternative='two-sided')
                stats_dict['mannwhitney_u_statistic'] = t_stat
                stats_dict['mannwhitney_p_value'] = p_value
                logger.info(f"  Mann-Whitney U test: U={t_stat:.2f}, p={p_value:.2e}")
                
                # Add to plot
                ax2.text(0.05, 0.95, f'Mann-Whitney U: p={p_value:.2e}',
                        transform=ax2.transAxes, fontsize=10,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            except Exception as e:
                logger.warning(f"Statistical test failed: {e}")
    else:
        # Box plot if no labels
        ax2.boxplot([spam_mass_values], labels=['All Nodes'])
        ax2.set_ylabel('Spam Mass', fontsize=12)
        ax2.set_title('Spam Mass Box Plot', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Distribution plot saved to {save_path}")
    else:
        plt.show()
    
    return stats_dict, fig


def identify_spam_by_spam_mass(
    spam_mass_dict: Dict[Union[int, str], float],
    pagerank_dict: Dict[Union[int, str], float],
    threshold: float = 0.7,
    min_pagerank: float = 0.0001
) -> Set[Union[int, str]]:
    """
    Identify suspected spam nodes based on spam mass and PageRank.
    
    Flags nodes that have:
    1. High spam mass (> threshold)
    2. Non-negligible PageRank (> min_pagerank)
    
    This filters out nodes with high spam mass but very low PageRank,
    which are likely just unimportant nodes rather than spam.
    
    Args:
        spam_mass_dict: Dictionary mapping node to spam mass
        pagerank_dict: Dictionary mapping node to PageRank score
        threshold: Spam mass threshold (default: 0.7)
        min_pagerank: Minimum PageRank to consider (default: 0.0001)
    
    Returns:
        Set of suspected spam node IDs
    """
    logger.info(f"Identifying spam nodes: threshold={threshold}, min_pagerank={min_pagerank}")
    
    suspected_spam = set()
    
    for node in spam_mass_dict.keys():
        spam_mass = spam_mass_dict.get(node, 0.0)
        pagerank = pagerank_dict.get(node, 0.0)
        
        if spam_mass > threshold and pagerank > min_pagerank:
            suspected_spam.add(node)
    
    logger.info(f"Identified {len(suspected_spam)} suspected spam nodes")
    logger.info(f"  With SM > {threshold} and PR > {min_pagerank}")
    
    return suspected_spam


def generate_spam_mass_report(
    pagerank: Dict[Union[int, str], float],
    trustrank: Dict[Union[int, str], float],
    spam_mass: Dict[Union[int, str], float],
    known_spam_nodes: Optional[Set[Union[int, str]]] = None,
    output_path: str = 'results/spam_mass_report.txt',
    threshold: float = 0.7
) -> None:
    """
    Generate comprehensive spam mass analysis report.
    
    Report includes:
    - Top-100 nodes by spam mass
    - Overall statistics
    - Classification summary
    - If known spam provided: confusion matrix, precision, recall, F1
    - Distribution statistics
    
    Args:
        pagerank: Dictionary of PageRank scores
        trustrank: Dictionary of TrustRank scores
        spam_mass: Dictionary of spam mass values
        known_spam_nodes: Optional set of known spam node IDs
        output_path: Path to save report
        threshold: Threshold for spam classification
    """
    logger.info(f"Generating spam mass report: {output_path}")
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Classify nodes
    classifications = classify_by_spam_mass(spam_mass, threshold=threshold)
    
    # Get top nodes by spam mass
    spam_mass_df = pd.DataFrame([
        {'node_id': node, 'spam_mass': sm, 'pagerank': pagerank.get(node, 0.0),
         'trustrank': trustrank.get(node, 0.0), 'classification': classifications.get(node, 'unknown')}
        for node, sm in spam_mass.items()
    ])
    spam_mass_df = spam_mass_df.sort_values('spam_mass', ascending=False)
    
    # Calculate statistics
    stats_dict, _ = analyze_spam_mass_distribution(spam_mass, true_spam_nodes=known_spam_nodes)
    
    # Write report
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SPAM MASS ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        
        # Summary statistics
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total nodes analyzed: {len(spam_mass):,}\n")
        f.write(f"Mean spam mass: {stats_dict['mean']:.6f}\n")
        f.write(f"Median spam mass: {stats_dict['median']:.6f}\n")
        f.write(f"Standard deviation: {stats_dict['std']:.6f}\n")
        f.write(f"95th percentile: {stats_dict['percentile_95']:.6f}\n")
        f.write(f"99th percentile: {stats_dict['percentile_99']:.6f}\n\n")
        
        # Classification summary
        f.write("CLASSIFICATION SUMMARY\n")
        f.write("-" * 80 + "\n")
        class_counts = pd.Series(classifications.values()).value_counts()
        for cls, count in class_counts.items():
            f.write(f"{cls.capitalize()}: {count:,} nodes ({100*count/len(classifications):.2f}%)\n")
        f.write("\n")
        
        # Top suspicious nodes
        f.write("TOP 100 NODES BY SPAM MASS\n")
        f.write("-" * 80 + "\n")
        top_100 = spam_mass_df.head(100)
        f.write(f"{'Rank':<6} {'Node ID':<15} {'Spam Mass':<12} {'PageRank':<12} "
               f"{'TrustRank':<12} {'Classification':<15}\n")
        f.write("-" * 80 + "\n")
        for idx, row in top_100.iterrows():
            rank = top_100.index.get_loc(idx) + 1
            f.write(f"{rank:<6} {str(row['node_id']):<15} {row['spam_mass']:<12.6f} "
                   f"{row['pagerank']:<12.6f} {row['trustrank']:<12.6f} {row['classification']:<15}\n")
        f.write("\n")
        
        # Evaluation metrics if known spam provided
        if known_spam_nodes:
            f.write("EVALUATION METRICS\n")
            f.write("-" * 80 + "\n")
            
            # Confusion matrix
            predicted_spam = {node for node, cls in classifications.items() if cls == 'likely_spam'}
            all_nodes = set(spam_mass.keys())
            true_spam = known_spam_nodes & all_nodes
            true_legitimate = all_nodes - known_spam_nodes
            
            tp = len(predicted_spam & true_spam)
            fp = len(predicted_spam & true_legitimate)
            fn = len(true_spam - predicted_spam)
            tn = len(true_legitimate - predicted_spam)
            
            f.write("Confusion Matrix:\n")
            f.write(f"                Predicted\n")
            f.write(f"              Spam  Legitimate\n")
            f.write(f"Actual Spam    {tp:4d}    {fn:4d}\n")
            f.write(f"Actual Legit   {fp:4d}    {tn:4d}\n\n")
            
            # Metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = (tp + tn) / len(all_nodes) if len(all_nodes) > 0 else 0.0
            
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1-Score: {f1_score:.4f}\n")
            f.write(f"Accuracy: {accuracy:.4f}\n\n")
            
            # Spam mass statistics for known spam vs legitimate
            spam_sm = [spam_mass[node] for node in true_spam if node in spam_mass]
            legit_sm = [spam_mass[node] for node in true_legitimate if node in spam_mass]
            
            if spam_sm and legit_sm:
                f.write("Spam Mass by True Label:\n")
                f.write(f"  Known Spam - Mean: {np.mean(spam_sm):.6f}, Median: {np.median(spam_sm):.6f}\n")
                f.write(f"  Legitimate - Mean: {np.mean(legit_sm):.6f}, Median: {np.median(legit_sm):.6f}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("End of Report\n")
        f.write("=" * 80 + "\n")
    
    logger.info(f"Report saved to {output_path}")


def compare_spam_mass_variants(
    G,
    pagerank: Dict[Union[int, str], float],
    trusted_sets_dict: Dict[str, Set[Union[int, str]]],
    known_spam_nodes: Optional[Set[Union[int, str]]] = None
) -> pd.DataFrame:
    """
    Compare spam mass detection using different trusted sets.
    
    Computes TrustRank and spam mass for each trusted set, then compares
    detection rates and performance metrics.
    
    Args:
        G: NetworkX graph
        pagerank: Dictionary of PageRank scores
        trusted_sets_dict: Dictionary mapping set_name to set of trusted nodes
        known_spam_nodes: Optional set of known spam node IDs for evaluation
    
    Returns:
        DataFrame with comparison metrics for each trusted set
    """
    logger.info(f"Comparing spam mass variants for {len(trusted_sets_dict)} trusted sets...")
    
    from trustrank import compute_trustrank
    
    results = []
    
    for set_name, trusted_nodes in tqdm(trusted_sets_dict.items(), desc="Computing variants"):
        try:
            # Compute TrustRank
            trustrank, _ = compute_trustrank(G, trusted_nodes, beta=0.85)
            
            # Calculate spam mass
            spam_mass = calculate_spam_mass(pagerank, trustrank)
            
            # Classify
            classifications = classify_by_spam_mass(spam_mass, threshold=0.7)
            likely_spam = {node for node, cls in classifications.items() if cls == 'likely_spam'}
            
            # Statistics
            sm_values = list(spam_mass.values())
            
            result = {
                'trusted_set': set_name,
                'num_trusted': len(trusted_nodes),
                'num_detected_spam': len(likely_spam),
                'detection_rate': len(likely_spam) / len(spam_mass) if spam_mass else 0.0,
                'mean_spam_mass': np.mean(sm_values) if sm_values else 0.0,
                'median_spam_mass': np.median(sm_values) if sm_values else 0.0,
                'p95_spam_mass': np.percentile(sm_values, 95) if sm_values else 0.0
            }
            
            # Evaluation metrics if known spam provided
            if known_spam_nodes:
                all_nodes = set(spam_mass.keys())
                true_spam = known_spam_nodes & all_nodes
                
                tp = len(likely_spam & true_spam)
                fp = len(likely_spam & (all_nodes - known_spam_nodes))
                fn = len(true_spam - likely_spam)
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                
                result['precision'] = precision
                result['recall'] = recall
                result['f1_score'] = f1_score
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"Failed to compute variant for {set_name}: {e}")
            continue
    
    comparison_df = pd.DataFrame(results)
    
    logger.info(f"Comparison completed: {len(comparison_df)} variants")
    if not comparison_df.empty:
        logger.info(f"  Best F1 (if evaluated): {comparison_df.get('f1_score', pd.Series([0])).max():.4f}")
    
    return comparison_df


def main():
    """
    Main function demonstrating spam mass calculation.
    """
    print("=" * 60)
    print("Spam Mass Module - Demo")
    print("=" * 60)
    
    try:
        import networkx as nx
        from trustrank import compute_trustrank, select_trusted_pages
        from centrality_analysis import compute_pagerank
        
        # Create a sample graph
        print("\n1. Creating sample graph...")
        G = nx.erdos_renyi_graph(n=500, p=0.01, seed=42)
        print(f"   Created graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Compute PageRank
        print("\n2. Computing PageRank...")
        pagerank = compute_pagerank(G, alpha=0.85, max_iter=100)
        print(f"   PageRank computed: {len(pagerank)} nodes")
        
        # Select trusted pages and compute TrustRank
        print("\n3. Computing TrustRank...")
        trusted = select_trusted_pages(G, method='top_pagerank', k=50)
        trustrank, _ = compute_trustrank(G, trusted, beta=0.85)
        print(f"   TrustRank computed: {len(trustrank)} nodes")
        
        # Calculate spam mass
        print("\n4. Calculating spam mass...")
        spam_mass = calculate_spam_mass(pagerank, trustrank)
        print(f"   Spam mass calculated: {len(spam_mass)} nodes")
        
        # Classify nodes
        print("\n5. Classifying nodes...")
        classifications = classify_by_spam_mass(spam_mass, threshold=0.7)
        print(f"   Classifications: {pd.Series(classifications.values()).value_counts().to_dict()}")
        
        # Analyze distribution
        print("\n6. Analyzing distribution...")
        stats_dict, fig = analyze_spam_mass_distribution(
            spam_mass,
            save_path='results/figures/spam_mass_distribution.png'
        )
        print(f"   Mean spam mass: {stats_dict['mean']:.4f}")
        print(f"   Median spam mass: {stats_dict['median']:.4f}")
        
        # Identify spam
        print("\n7. Identifying suspected spam...")
        suspected_spam = identify_spam_by_spam_mass(spam_mass, pagerank, threshold=0.7)
        print(f"   Suspected spam nodes: {len(suspected_spam)}")
        
        # Generate report
        print("\n8. Generating report...")
        generate_spam_mass_report(
            pagerank, trustrank, spam_mass,
            output_path='results/spam_mass_report.txt'
        )
        print("   ✅ Report saved")
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)
    
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"\n✗ Error: {e}")
        raise


if __name__ == "__main__":
    main()

