"""
Spam detection evaluation module for network analysis.

This module provides comprehensive evaluation tools for comparing different
spam detection methods, including metrics calculation, visualization, and
statistical analysis.

Evaluation capabilities:
1. Confusion matrix and standard metrics (Precision, Recall, F1, Accuracy)
2. Method comparison and ranking
3. ROC and Precision-Recall curves
4. Detection overlap analysis
5. False positive/negative analysis
6. Comprehensive reporting

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
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score, accuracy_score,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    roc_auc_score
)

# Try to import matplotlib_venn for overlap visualization
try:
    from matplotlib_venn import venn2, venn3, venn2_circles, venn3_circles
    HAS_VENN = True
except ImportError:
    HAS_VENN = False
    logger = logging.getLogger(__name__)
    logger.warning("matplotlib_venn not available, using alternative visualization")

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


def evaluate_detection_method(
    detected_spam: Set[Union[int, str]],
    true_spam: Set[Union[int, str]],
    all_nodes: Optional[Set[Union[int, str]]] = None
) -> Tuple[Dict, np.ndarray]:
    """
    Evaluate a single spam detection method.
    
    Calculates confusion matrix and standard classification metrics.
    
    Args:
        detected_spam: Set of nodes detected as spam
        true_spam: Set of nodes that are actually spam
        all_nodes: Optional set of all nodes in graph (if None, uses union of detected and true)
    
    Returns:
        Tuple of (metrics_dict, confusion_matrix_array)
        metrics_dict contains: precision, recall, f1_score, accuracy, fpr, tp, fp, tn, fn
    """
    logger.info("Evaluating detection method...")
    
    # Determine all nodes
    if all_nodes is None:
        all_nodes = detected_spam | true_spam
    
    # Calculate confusion matrix components
    tp = len(detected_spam & true_spam)  # True Positives
    fp = len(detected_spam - true_spam)   # False Positives
    fn = len(true_spam - detected_spam)  # False Negatives
    tn = len(all_nodes - detected_spam - true_spam)  # True Negatives
    
    # Create confusion matrix
    cm = np.array([[tn, fp],
                   [fn, tp]])
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(all_nodes) if len(all_nodes) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy,
        'false_positive_rate': fpr,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn,
        'total_detected': len(detected_spam),
        'total_spam': len(true_spam),
        'total_nodes': len(all_nodes)
    }
    
    logger.info(f"Evaluation results:")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1-Score: {f1:.4f}")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    
    return metrics, cm


def compare_detection_methods(
    true_spam: Set[Union[int, str]],
    detection_results_dict: Dict[str, Set[Union[int, str]]],
    all_nodes: Optional[Set[Union[int, str]]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare multiple spam detection methods.
    
    Evaluates each method and creates a comparison table with all metrics.
    Methods are ranked by F1-score.
    
    Args:
        true_spam: Set of nodes that are actually spam
        detection_results_dict: Dictionary mapping method_name to set of detected nodes
        all_nodes: Optional set of all nodes in graph
    
    Returns:
        Tuple of (comparison_dataframe, rankings_dataframe)
    """
    logger.info(f"Comparing {len(detection_results_dict)} detection methods...")
    
    if all_nodes is None:
        # Union of all detected nodes and true spam
        all_nodes = true_spam.copy()
        for detected in detection_results_dict.values():
            all_nodes.update(detected)
    
    results = []
    
    for method_name, detected_spam in tqdm(detection_results_dict.items(), desc="Evaluating methods"):
        metrics, _ = evaluate_detection_method(detected_spam, true_spam, all_nodes)
        
        result = {
            'method': method_name,
            **metrics
        }
        results.append(result)
    
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.sort_values('f1_score', ascending=False)
    
    # Create rankings
    rankings = comparison_df[['method', 'f1_score', 'precision', 'recall', 'accuracy']].copy()
    rankings['rank'] = range(1, len(rankings) + 1)
    rankings = rankings[['rank', 'method', 'f1_score', 'precision', 'recall', 'accuracy']]
    
    logger.info(f"Comparison completed: {len(comparison_df)} methods")
    logger.info(f"  Best F1-Score: {comparison_df['f1_score'].max():.4f} ({comparison_df.iloc[0]['method']})")
    
    return comparison_df, rankings


def plot_roc_curves(
    true_spam: Set[Union[int, str]],
    detection_scores_dict: Dict[str, Dict[Union[int, str], float]],
    all_nodes: Optional[Set[Union[int, str]]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> Dict[str, float]:
    """
    Plot ROC curves for methods that produce scores.
    
    For each method, calculates ROC curve at different thresholds and plots.
    Also calculates Area Under Curve (AUC) for each method.
    
    Args:
        true_spam: Set of nodes that are actually spam
        detection_scores_dict: Dictionary mapping method_name to {node: score}
        all_nodes: Optional set of all nodes (default: union of all scored nodes)
        save_path: Optional path to save figure
        figsize: Figure size tuple (default: (10, 8))
    
    Returns:
        Dictionary mapping method_name to AUC score
    """
    logger.info("Plotting ROC curves...")
    
    if all_nodes is None:
        all_nodes = true_spam.copy()
        for scores in detection_scores_dict.values():
            all_nodes.update(scores.keys())
    
    fig, ax = plt.subplots(figsize=figsize)
    
    auc_scores = {}
    
    for method_name, scores in detection_scores_dict.items():
        # Prepare labels and scores
        y_true = []
        y_scores = []
        
        for node in all_nodes:
            if node in scores:
                y_true.append(1 if node in true_spam else 0)
                y_scores.append(scores[node])
        
        if not y_true or sum(y_true) == 0:
            logger.warning(f"No positive samples for {method_name}, skipping ROC curve")
            continue
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        auc_scores[method_name] = roc_auc
        
        # Plot
        ax.plot(fpr, tpr, linewidth=2, label=f'{method_name} (AUC = {roc_auc:.3f})')
    
    # Plot diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier (AUC = 0.500)')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - Spam Detection Methods', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC curves saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    logger.info(f"AUC scores: {auc_scores}")
    
    return auc_scores


def plot_precision_recall_curves(
    true_spam: Set[Union[int, str]],
    detection_scores_dict: Dict[str, Dict[Union[int, str], float]],
    all_nodes: Optional[Set[Union[int, str]]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> Dict[str, float]:
    """
    Plot Precision-Recall curves for methods that produce scores.
    
    Precision-Recall curves are better for imbalanced datasets than ROC curves.
    Calculates Average Precision (AP) for each method.
    
    Args:
        true_spam: Set of nodes that are actually spam
        detection_scores_dict: Dictionary mapping method_name to {node: score}
        all_nodes: Optional set of all nodes
        save_path: Optional path to save figure
        figsize: Figure size tuple (default: (10, 8))
    
    Returns:
        Dictionary mapping method_name to Average Precision score
    """
    logger.info("Plotting Precision-Recall curves...")
    
    if all_nodes is None:
        all_nodes = true_spam.copy()
        for scores in detection_scores_dict.values():
            all_nodes.update(scores.keys())
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ap_scores = {}
    
    for method_name, scores in detection_scores_dict.items():
        # Prepare labels and scores
        y_true = []
        y_scores = []
        
        for node in all_nodes:
            if node in scores:
                y_true.append(1 if node in true_spam else 0)
                y_scores.append(scores[node])
        
        if not y_true or sum(y_true) == 0:
            logger.warning(f"No positive samples for {method_name}, skipping PR curve")
            continue
        
        # Calculate Precision-Recall curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
        ap_scores[method_name] = ap
        
        # Plot
        ax.plot(recall, precision, linewidth=2, label=f'{method_name} (AP = {ap:.3f})')
    
    # Baseline (random classifier)
    baseline = len(true_spam & all_nodes) / len(all_nodes) if all_nodes else 0.0
    ax.axhline(y=baseline, color='k', linestyle='--', linewidth=1, 
              label=f'Random Classifier (AP = {baseline:.3f})')
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves - Spam Detection Methods', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Precision-Recall curves saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    logger.info(f"Average Precision scores: {ap_scores}")
    
    return ap_scores


def analyze_detection_overlap(
    detection_results_dict: Dict[str, Set[Union[int, str]]],
    true_spam: Optional[Set[Union[int, str]]] = None
) -> Tuple[Dict, plt.Figure]:
    """
    Analyze overlap between different detection methods.
    
    Creates Venn diagrams showing which spam is caught by which methods.
    Also identifies spam caught by all methods and spam missed by all methods.
    
    Args:
        detection_results_dict: Dictionary mapping method_name to set of detected nodes
        true_spam: Optional set of true spam nodes for validation
    
    Returns:
        Tuple of (overlap_statistics_dict, figure)
    """
    logger.info("Analyzing detection overlap...")
    
    methods = list(detection_results_dict.keys())
    detected_sets = list(detection_results_dict.values())
    
    # Calculate overlap statistics
    stats_dict = {}
    
    # All detected nodes
    all_detected = set()
    for detected in detected_sets:
        all_detected.update(detected)
    stats_dict['total_detected'] = len(all_detected)
    
    # Nodes detected by all methods
    if detected_sets:
        detected_by_all = set.intersection(*detected_sets)
        stats_dict['detected_by_all'] = len(detected_by_all)
        stats_dict['detected_by_all_nodes'] = detected_by_all
    else:
        detected_by_all = set()
        stats_dict['detected_by_all'] = 0
        stats_dict['detected_by_all_nodes'] = set()
    
    # Nodes detected by only one method
    detected_by_one = set()
    for i, method_set in enumerate(detected_sets):
        others = set()
        for j, other_set in enumerate(detected_sets):
            if i != j:
                others.update(other_set)
        only_this = method_set - others
        detected_by_one.update(only_this)
        stats_dict[f'only_{methods[i]}'] = len(only_this)
    
    stats_dict['detected_by_one_only'] = len(detected_by_one)
    
    # If true spam provided, calculate additional stats
    if true_spam:
        stats_dict['true_spam_total'] = len(true_spam)
        stats_dict['true_spam_detected_by_all'] = len(detected_by_all & true_spam)
        stats_dict['true_spam_missed_by_all'] = len(true_spam - all_detected)
        
        # Calculate per-method recall
        for method_name, detected in detection_results_dict.items():
            recall = len(detected & true_spam) / len(true_spam) if true_spam else 0.0
            stats_dict[f'{method_name}_recall'] = recall
    
    # Create visualization
    fig = None
    
    if len(methods) == 2 and HAS_VENN:
        # Two-method Venn diagram
        fig, ax = plt.subplots(figsize=(10, 8))
        venn2([detected_sets[0], detected_sets[1]], 
              set_labels=(methods[0], methods[1]), ax=ax)
        ax.set_title('Detection Overlap: 2 Methods', fontsize=14, fontweight='bold')
        plt.tight_layout()
    
    elif len(methods) == 3 and HAS_VENN:
        # Three-method Venn diagram
        fig, ax = plt.subplots(figsize=(12, 10))
        venn3([detected_sets[0], detected_sets[1], detected_sets[2]],
              set_labels=(methods[0], methods[1], methods[2]), ax=ax)
        ax.set_title('Detection Overlap: 3 Methods', fontsize=14, fontweight='bold')
        plt.tight_layout()
    
    else:
        # For more than 3 methods, create a heatmap
        fig, ax = plt.subplots(figsize=(max(8, len(methods)), max(6, len(methods))))
        
        # Create overlap matrix
        overlap_matrix = np.zeros((len(methods), len(methods)))
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                if i == j:
                    overlap_matrix[i, j] = len(detection_results_dict[method1])
                else:
                    overlap = len(detection_results_dict[method1] & detection_results_dict[method2])
                    overlap_matrix[i, j] = overlap
        
        sns.heatmap(overlap_matrix, annot=True, fmt='.0f', cmap='YlOrRd',
                    xticklabels=methods, yticklabels=methods, ax=ax)
        ax.set_title('Detection Overlap Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
    
    logger.info(f"Overlap analysis completed:")
    logger.info(f"  Total detected: {stats_dict['total_detected']}")
    logger.info(f"  Detected by all: {stats_dict['detected_by_all']}")
    logger.info(f"  Detected by one only: {stats_dict['detected_by_one_only']}")
    
    return stats_dict, fig


def analyze_false_positives(
    false_positives_by_method: Dict[str, Set[Union[int, str]]],
    G,
    pagerank: Optional[Dict[Union[int, str], float]] = None
) -> Dict:
    """
    Analyze characteristics of false positive detections.
    
    Examines what patterns false positives have - are they high PageRank
    legitimate nodes? What structural properties do they share?
    
    Args:
        false_positives_by_method: Dictionary mapping method_name to set of false positives
        G: NetworkX graph
        pagerank: Optional PageRank scores for nodes
    
    Returns:
        Dictionary with analysis results
    """
    logger.info("Analyzing false positives...")
    
    analysis = {}
    
    for method_name, fps in false_positives_by_method.items():
        if not fps:
            continue
        
        method_analysis = {
            'count': len(fps),
            'avg_degree': np.mean([G.degree(n) for n in fps]) if fps else 0.0,
            'avg_clustering': np.mean([nx.clustering(G, n) for n in fps]) if fps else 0.0,
        }
        
        if pagerank:
            fp_pageranks = [pagerank.get(n, 0.0) for n in fps if n in pagerank]
            if fp_pageranks:
                method_analysis['avg_pagerank'] = np.mean(fp_pageranks)
                method_analysis['max_pagerank'] = max(fp_pageranks)
                method_analysis['high_pagerank_count'] = sum(1 for pr in fp_pageranks if pr > np.percentile(list(pagerank.values()), 90))
        
        analysis[method_name] = method_analysis
    
    logger.info(f"False positive analysis completed for {len(analysis)} methods")
    
    return analysis


def analyze_false_negatives(
    false_negatives_by_method: Dict[str, Set[Union[int, str]]],
    G,
    spam_metadata: Optional[Dict] = None
) -> Dict:
    """
    Analyze which spam farms evaded detection.
    
    Examines characteristics of spam that was not detected - what patterns
    allow them to evade detection?
    
    Args:
        false_negatives_by_method: Dictionary mapping method_name to set of false negatives
        G: NetworkX graph
        spam_metadata: Optional metadata about spam farms (structure, type, etc.)
    
    Returns:
        Dictionary with analysis results
    """
    logger.info("Analyzing false negatives...")
    
    analysis = {}
    
    for method_name, fns in false_negatives_by_method.items():
        if not fns:
            continue
        
        method_analysis = {
            'count': len(fns),
            'avg_degree': np.mean([G.degree(n) for n in fns]) if fns else 0.0,
            'avg_clustering': np.mean([nx.clustering(G, n) for n in fns]) if fns else 0.0,
        }
        
        if spam_metadata:
            # Check if false negatives are from specific spam farm types
            for fn in fns:
                if fn in spam_metadata:
                    farm_type = spam_metadata[fn].get('farm_type', 'unknown')
                    if 'farm_types' not in method_analysis:
                        method_analysis['farm_types'] = defaultdict(int)
                    method_analysis['farm_types'][farm_type] += 1
        
        analysis[method_name] = method_analysis
    
    logger.info(f"False negative analysis completed for {len(analysis)} methods")
    
    return analysis


def generate_detection_report(
    all_results: Dict,
    output_dir: str,
    include_plots: bool = True
) -> str:
    """
    Generate comprehensive HTML report with all evaluation results.
    
    Report includes:
    - Method comparison table
    - ROC and PR curves
    - Confusion matrices
    - Best/worst performers
    - Recommendations
    
    Args:
        all_results: Dictionary containing all evaluation results
        output_dir: Directory to save report
        include_plots: Whether to embed plots in HTML (default: True)
    
    Returns:
        Path to generated HTML report
    """
    logger.info("Generating comprehensive detection report...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create HTML report
    html_path = os.path.join(output_dir, 'spam_detection_report.html')
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Spam Detection Evaluation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #34495e; margin-top: 30px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #3498db; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .metric {{ font-weight: bold; color: #27ae60; }}
            .warning {{ color: #e74c3c; }}
            img {{ max-width: 100%; height: auto; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>Spam Detection Evaluation Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Executive Summary</h2>
        <p>This report evaluates {len(all_results.get('comparison_df', []))} spam detection methods
        on the network analysis task.</p>
    """
    
    # Add comparison table
    if 'comparison_df' in all_results and not all_results['comparison_df'].empty:
        html_content += "<h2>Method Comparison</h2>\n"
        html_content += all_results['comparison_df'].to_html(index=False, classes='comparison-table')
        html_content += "\n"
        
        # Best performer
        best = all_results['comparison_df'].iloc[0]
        html_content += f"""
        <h3>Best Performer</h3>
        <p><strong>{best['method']}</strong> achieved the highest F1-Score of <span class="metric">{best['f1_score']:.4f}</span>
        (Precision: {best['precision']:.4f}, Recall: {best['recall']:.4f})</p>
        """
    
    # Add ROC curves
    if include_plots and 'roc_curves_path' in all_results:
        html_content += f"""
        <h2>ROC Curves</h2>
        <img src="{os.path.basename(all_results['roc_curves_path'])}" alt="ROC Curves">
        """
    
    # Add PR curves
    if include_plots and 'pr_curves_path' in all_results:
        html_content += f"""
        <h2>Precision-Recall Curves</h2>
        <img src="{os.path.basename(all_results['pr_curves_path'])}" alt="Precision-Recall Curves">
        """
    
    # Add recommendations
    html_content += """
        <h2>Recommendations</h2>
        <ul>
            <li>Use ensemble methods combining multiple detection approaches</li>
            <li>Focus on methods with high precision to minimize false positives</li>
            <li>Consider domain-specific adjustments based on network characteristics</li>
            <li>Regularly re-evaluate as spam patterns evolve</li>
        </ul>
    """
    
    html_content += """
    </body>
    </html>
    """
    
    # Write HTML file
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Report saved to {html_path}")
    
    return html_path


def main():
    """
    Main function demonstrating spam detection evaluation.
    """
    print("=" * 60)
    print("Spam Detection Evaluation Module - Demo")
    print("=" * 60)
    
    try:
        import networkx as nx
        
        # Create sample data
        print("\n1. Creating sample detection results...")
        all_nodes = set(range(1000))
        true_spam = set(range(50, 150))  # Nodes 50-149 are spam
        
        # Simulate detection results
        method1_detected = set(range(40, 140))  # Some overlap
        method2_detected = set(range(60, 160))  # Different overlap
        method3_detected = set(range(100, 200))  # Different again
        
        detection_results = {
            'method1': method1_detected,
            'method2': method2_detected,
            'method3': method3_detected
        }
        
        print(f"   True spam: {len(true_spam)} nodes")
        print(f"   Method detections: {[len(d) for d in detection_results.values()]}")
        
        # Evaluate single method
        print("\n2. Evaluating single method...")
        metrics, cm = evaluate_detection_method(method1_detected, true_spam, all_nodes)
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall: {metrics['recall']:.4f}")
        print(f"   F1-Score: {metrics['f1_score']:.4f}")
        
        # Compare methods
        print("\n3. Comparing all methods...")
        comparison_df, rankings = compare_detection_methods(true_spam, detection_results, all_nodes)
        print(comparison_df[['method', 'precision', 'recall', 'f1_score']].to_string(index=False))
        
        # Analyze overlap
        print("\n4. Analyzing detection overlap...")
        overlap_stats, fig = analyze_detection_overlap(detection_results, true_spam)
        print(f"   Detected by all: {overlap_stats['detected_by_all']}")
        print(f"   Detected by one only: {overlap_stats['detected_by_one_only']}")
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)
    
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"\n✗ Error: {e}")
        raise


if __name__ == "__main__":
    main()

