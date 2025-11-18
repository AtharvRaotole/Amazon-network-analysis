"""
Link prediction evaluation and visualization module.

This module provides functions for visualizing and comparing link prediction
results, including ROC curves, precision-recall curves, confusion matrices,
and comprehensive reports.
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

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


def plot_roc_curves(
    results_dict: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Plot ROC curves for multiple link prediction methods.
    
    Args:
        results_dict: Dictionary with method names as keys and tuples of
                     (fpr, tpr, auc) as values
        save_path: Optional path to save the figure
        figsize: Figure size tuple (default: (10, 8))
    
    Raises:
        ValueError: If results_dict is empty
    """
    if not results_dict:
        raise ValueError("results_dict is empty")
    
    logger.info(f"Plotting ROC curves for {len(results_dict)} methods...")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot ROC curve for each method
    for method_name, (fpr, tpr, auc) in results_dict.items():
        ax.plot(fpr, tpr, linewidth=2, label=f'{method_name} (AUC = {auc:.4f})')
    
    # Plot diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier (AUC = 0.5000)')
    
    # Formatting
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - Link Prediction Methods', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC curves saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_precision_recall_curves(
    results_dict: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Plot Precision-Recall curves for multiple link prediction methods.
    
    Args:
        results_dict: Dictionary with method names as keys and tuples of
                     (precision, recall, ap) as values
        save_path: Optional path to save the figure
        figsize: Figure size tuple (default: (10, 8))
    
    Raises:
        ValueError: If results_dict is empty
    """
    if not results_dict:
        raise ValueError("results_dict is empty")
    
    logger.info(f"Plotting Precision-Recall curves for {len(results_dict)} methods...")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot PR curve for each method
    for method_name, (precision, recall, ap) in results_dict.items():
        ax.plot(recall, precision, linewidth=2, label=f'{method_name} (AP = {ap:.4f})')
    
    # Formatting
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves - Link Prediction Methods', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Precision-Recall curves saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrices(
    results_dict: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10)
) -> None:
    """
    Plot confusion matrices for multiple link prediction methods.
    
    Args:
        results_dict: Dictionary with method names as keys and confusion
                     matrices (2x2 numpy arrays) as values
        save_path: Optional path to save the figure
        figsize: Figure size tuple (default: (12, 10))
    
    Raises:
        ValueError: If results_dict is empty
    """
    if not results_dict:
        raise ValueError("results_dict is empty")
    
    logger.info(f"Plotting confusion matrices for {len(results_dict)} methods...")
    
    n_methods = len(results_dict)
    n_cols = 2
    n_rows = (n_methods + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_methods == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (method_name, cm) in enumerate(results_dict.items()):
        ax = axes[idx]
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2%',
            cmap='Blues',
            cbar=True,
            ax=ax,
            square=True,
            linewidths=0.5
        )
        
        # Add raw counts as text
        for i in range(2):
            for j in range(2):
                ax.text(j + 0.5, i + 0.7, f'({int(cm[i, j])})',
                       ha='center', va='center', fontsize=9, color='red', fontweight='bold')
        
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('Actual', fontsize=10)
        ax.set_title(f'{method_name}', fontsize=11, fontweight='bold')
        ax.set_xticklabels(['No Link', 'Link'])
        ax.set_yticklabels(['No Link', 'Link'])
    
    # Hide unused subplots
    for idx in range(n_methods, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Confusion Matrices - Link Prediction Methods', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrices saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_feature_importance(
    importance_df: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    top_k: Optional[int] = None
) -> None:
    """
    Plot feature importances from Random Forest model.
    
    Args:
        importance_df: DataFrame with columns 'feature' and 'importance'
        save_path: Optional path to save the figure
        figsize: Figure size tuple (default: (10, 6))
        top_k: Optional number of top features to display (default: all)
    
    Raises:
        ValueError: If importance_df is empty or missing required columns
    """
    if importance_df.empty:
        raise ValueError("importance_df is empty")
    
    if 'feature' not in importance_df.columns or 'importance' not in importance_df.columns:
        raise ValueError("importance_df must have 'feature' and 'importance' columns")
    
    logger.info(f"Plotting feature importances...")
    
    # Select top k if specified
    df_plot = importance_df.head(top_k) if top_k else importance_df
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create horizontal bar chart
    y_pos = np.arange(len(df_plot))
    colors = plt.cm.viridis(np.linspace(0, 1, len(df_plot)))
    
    bars = ax.barh(y_pos, df_plot['importance'].values, color=colors)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, df_plot['importance'].values)):
        ax.text(val + 0.01 * max(df_plot['importance']), i, f'{val:.4f}',
               va='center', fontsize=9)
    
    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot['feature'].values)
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title('Feature Importances - Random Forest Model', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()  # Highest importance at top
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_comparison_table(
    all_results: Dict[str, Dict[str, float]]
) -> pd.DataFrame:
    """
    Create a comparison table of metrics for all link prediction methods.
    
    Args:
        all_results: Dictionary with method names as keys and dictionaries
                    of metrics as values. Metrics should include:
                    'precision', 'recall', 'f1', 'auc_roc', 'auc_pr'
    
    Returns:
        Styled DataFrame with comparison metrics
    
    Raises:
        ValueError: If all_results is empty
    """
    if not all_results:
        raise ValueError("all_results is empty")
    
    logger.info(f"Creating comparison table for {len(all_results)} methods...")
    
    # Extract metrics for each method
    rows = []
    for method_name, metrics in all_results.items():
        row = {
            'Method': method_name,
            'Precision': metrics.get('precision', np.nan),
            'Recall': metrics.get('recall', np.nan),
            'F1': metrics.get('f1', np.nan),
            'AUC-ROC': metrics.get('auc_roc', np.nan),
            'AUC-PR': metrics.get('auc_pr', np.nan)
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort by F1 score descending
    df = df.sort_values('F1', ascending=False)
    
    # Style the DataFrame
    styled_df = df.style.format({
        'Precision': '{:.4f}',
        'Recall': '{:.4f}',
        'F1': '{:.4f}',
        'AUC-ROC': '{:.4f}',
        'AUC-PR': '{:.4f}'
    }).background_gradient(subset=['Precision', 'Recall', 'F1', 'AUC-ROC', 'AUC-PR'], cmap='YlOrRd')
    
    logger.info("Comparison table created")
    
    return df, styled_df


def plot_score_distributions(
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
    method_name: str,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    bins: int = 50
) -> None:
    """
    Plot score distributions for positive and negative edges.
    
    Args:
        pos_scores: Array of prediction scores for positive edges
        neg_scores: Array of prediction scores for negative edges
        method_name: Name of the prediction method
        save_path: Optional path to save the figure
        figsize: Figure size tuple (default: (10, 6))
        bins: Number of bins for histogram (default: 50)
    
    Raises:
        ValueError: If arrays are empty
    """
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        raise ValueError("pos_scores and neg_scores must not be empty")
    
    logger.info(f"Plotting score distributions for {method_name}...")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot histograms
    ax.hist(neg_scores, bins=bins, alpha=0.6, label='Negative Edges', color='red', edgecolor='black')
    ax.hist(pos_scores, bins=bins, alpha=0.6, label='Positive Edges', color='green', edgecolor='black')
    
    # Add vertical lines for means
    mean_neg = np.mean(neg_scores)
    mean_pos = np.mean(pos_scores)
    ax.axvline(mean_neg, color='red', linestyle='--', linewidth=2, label=f'Mean Negative: {mean_neg:.4f}')
    ax.axvline(mean_pos, color='green', linestyle='--', linewidth=2, label=f'Mean Positive: {mean_pos:.4f}')
    
    # Formatting
    ax.set_xlabel('Prediction Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Score Distributions - {method_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Score distribution plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def generate_link_prediction_report(
    all_results: Dict[str, Dict],
    output_path: str,
    roc_data: Optional[Dict[str, Tuple]] = None,
    pr_data: Optional[Dict[str, Tuple]] = None,
    confusion_matrices: Optional[Dict[str, np.ndarray]] = None,
    feature_importance: Optional[pd.DataFrame] = None,
    score_distributions: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None
) -> None:
    """
    Generate comprehensive HTML report with all metrics and plots.
    
    Args:
        all_results: Dictionary with method names and their metrics
        output_path: Path to save HTML report
        roc_data: Optional ROC curve data (method_name: (fpr, tpr, auc))
        pr_data: Optional PR curve data (method_name: (precision, recall, ap))
        confusion_matrices: Optional confusion matrices (method_name: cm)
        feature_importance: Optional feature importance DataFrame
        score_distributions: Optional score distributions (method_name: (pos_scores, neg_scores))
    
    Raises:
        ValueError: If all_results is empty
    """
    if not all_results:
        raise ValueError("all_results is empty")
    
    logger.info(f"Generating comprehensive link prediction report...")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Create comparison table
    comparison_df, styled_df = create_comparison_table(all_results)
    
    # Generate plots and save them
    plot_dir = os.path.join(os.path.dirname(output_path), 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    plot_paths = {}
    
    if roc_data:
        roc_path = os.path.join(plot_dir, 'roc_curves.png')
        plot_roc_curves(roc_data, save_path=roc_path)
        plot_paths['roc'] = roc_path
    
    if pr_data:
        pr_path = os.path.join(plot_dir, 'precision_recall_curves.png')
        plot_precision_recall_curves(pr_data, save_path=pr_path)
        plot_paths['pr'] = pr_path
    
    if confusion_matrices:
        cm_path = os.path.join(plot_dir, 'confusion_matrices.png')
        plot_confusion_matrices(confusion_matrices, save_path=cm_path)
        plot_paths['cm'] = cm_path
    
    if feature_importance is not None:
        fi_path = os.path.join(plot_dir, 'feature_importance.png')
        plot_feature_importance(feature_importance, save_path=fi_path)
        plot_paths['fi'] = fi_path
    
    if score_distributions:
        for method_name, (pos_scores, neg_scores) in score_distributions.items():
            dist_path = os.path.join(plot_dir, f'score_distribution_{method_name.replace(" ", "_")}.png')
            plot_score_distributions(pos_scores, neg_scores, method_name, save_path=dist_path)
            plot_paths[f'dist_{method_name}'] = dist_path
    
    # Generate HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Link Prediction Evaluation Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #34495e;
                margin-top: 30px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #3498db;
                color: white;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            img {{
                max-width: 100%;
                height: auto;
                margin: 20px 0;
                border: 1px solid #ddd;
                border-radius: 5px;
            }}
            .plot-section {{
                margin: 30px 0;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Link Prediction Evaluation Report</h1>
            
            <h2>1. Comparison Table</h2>
            {comparison_df.to_html(index=False, classes='table', escape=False)}
            
            <h2>2. ROC Curves</h2>
            <div class="plot-section">
                {'<img src="' + os.path.basename(plot_paths['roc']) + '" alt="ROC Curves">' if 'roc' in plot_paths else '<p>ROC curves not available</p>'}
            </div>
            
            <h2>3. Precision-Recall Curves</h2>
            <div class="plot-section">
                {'<img src="' + os.path.basename(plot_paths['pr']) + '" alt="PR Curves">' if 'pr' in plot_paths else '<p>PR curves not available</p>'}
            </div>
            
            <h2>4. Confusion Matrices</h2>
            <div class="plot-section">
                {'<img src="' + os.path.basename(plot_paths['cm']) + '" alt="Confusion Matrices">' if 'cm' in plot_paths else '<p>Confusion matrices not available</p>'}
            </div>
            
            <h2>5. Feature Importance</h2>
            <div class="plot-section">
                {'<img src="' + os.path.basename(plot_paths['fi']) + '" alt="Feature Importance">' if 'fi' in plot_paths else '<p>Feature importance not available</p>'}
            </div>
            
            <h2>6. Score Distributions</h2>
            <div class="plot-section">
    """
    
    if score_distributions:
        for method_name in score_distributions.keys():
            key = f'dist_{method_name}'
            if key in plot_paths:
                html_content += f'<h3>{method_name}</h3>'
                html_content += f'<img src="{os.path.basename(plot_paths[key])}" alt="Score Distribution - {method_name}">'
    else:
        html_content += '<p>Score distributions not available</p>'
    
    html_content += """
            </div>
            
            <h2>7. Detailed Metrics</h2>
            <table>
                <tr>
                    <th>Method</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1 Score</th>
                    <th>AUC-ROC</th>
                    <th>AUC-PR</th>
                </tr>
    """
    
    for method_name, metrics in all_results.items():
        precision = f"{metrics.get('precision', 0):.4f}" if isinstance(metrics.get('precision'), (int, float)) else 'N/A'
        recall = f"{metrics.get('recall', 0):.4f}" if isinstance(metrics.get('recall'), (int, float)) else 'N/A'
        f1 = f"{metrics.get('f1', 0):.4f}" if isinstance(metrics.get('f1'), (int, float)) else 'N/A'
        auc_roc = f"{metrics.get('auc_roc', 0):.4f}" if isinstance(metrics.get('auc_roc'), (int, float)) else 'N/A'
        auc_pr = f"{metrics.get('auc_pr', 0):.4f}" if isinstance(metrics.get('auc_pr'), (int, float)) else 'N/A'
        
        html_content += f"""
                <tr>
                    <td>{method_name}</td>
                    <td>{precision}</td>
                    <td>{recall}</td>
                    <td>{f1}</td>
                    <td>{auc_roc}</td>
                    <td>{auc_pr}</td>
                </tr>
        """
    
    html_content += """
            </table>
        </div>
    </body>
    </html>
    """
    
    # Write HTML file
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Comprehensive report saved to {output_path}")
    
    # Also save comparison table as CSV
    csv_path = output_path.replace('.html', '_comparison.csv')
    comparison_df.to_csv(csv_path, index=False)
    logger.info(f"Comparison table saved to {csv_path}")


def main():
    """
    Main function demonstrating usage of the link prediction evaluation module.
    """
    print("=" * 60)
    print("Link Prediction Evaluation Module - Demo")
    print("=" * 60)
    
    try:
        import numpy as np
        from sklearn.metrics import roc_curve, precision_recall_curve
        
        # Create sample data
        print("\n1. Creating sample evaluation data...")
        
        # Sample metrics
        all_results = {
            'Adamic-Adar': {
                'precision': 0.75,
                'recall': 0.70,
                'f1': 0.72,
                'auc_roc': 0.85,
                'auc_pr': 0.80
            },
            'Random Forest': {
                'precision': 0.90,
                'recall': 0.88,
                'f1': 0.89,
                'auc_roc': 0.95,
                'auc_pr': 0.92
            }
        }
        
        # Sample ROC data
        np.random.seed(42)
        fpr1, tpr1 = np.linspace(0, 1, 100), np.linspace(0, 1, 100) ** 0.5
        fpr2, tpr2 = np.linspace(0, 1, 100), np.linspace(0, 1, 100) ** 0.7
        
        roc_data = {
            'Adamic-Adar': (fpr1, tpr1, 0.85),
            'Random Forest': (fpr2, tpr2, 0.95)
        }
        
        # Sample PR data
        recall1, precision1 = np.linspace(0, 1, 100), np.linspace(1, 0.7, 100)
        recall2, precision2 = np.linspace(0, 1, 100), np.linspace(1, 0.85, 100)
        
        pr_data = {
            'Adamic-Adar': (precision1, recall1, 0.80),
            'Random Forest': (precision2, recall2, 0.92)
        }
        
        # Sample confusion matrices
        confusion_matrices = {
            'Adamic-Adar': np.array([[80, 20], [30, 70]]),
            'Random Forest': np.array([[90, 10], [12, 88]])
        }
        
        # Sample feature importance
        feature_importance = pd.DataFrame({
            'feature': ['adamic_adar', 'common_neighbors', 'jaccard_coefficient',
                       'degree_u', 'degree_v', 'clustering_u'],
            'importance': [0.35, 0.25, 0.15, 0.10, 0.08, 0.07]
        }).sort_values('importance', ascending=False)
        
        # Sample score distributions
        pos_scores1 = np.random.beta(2, 1, 1000)
        neg_scores1 = np.random.beta(1, 2, 1000)
        pos_scores2 = np.random.beta(3, 1, 1000)
        neg_scores2 = np.random.beta(1, 3, 1000)
        
        score_distributions = {
            'Adamic-Adar': (pos_scores1, neg_scores1),
            'Random Forest': (pos_scores2, neg_scores2)
        }
        
        print("   ✅ Sample data created")
        
        # Test plotting functions
        print("\n2. Testing plotting functions...")
        
        plot_roc_curves(roc_data, save_path='results/figures/test_roc.png')
        print("   ✅ ROC curves plotted")
        
        plot_precision_recall_curves(pr_data, save_path='results/figures/test_pr.png')
        print("   ✅ Precision-Recall curves plotted")
        
        plot_confusion_matrices(confusion_matrices, save_path='results/figures/test_cm.png')
        print("   ✅ Confusion matrices plotted")
        
        plot_feature_importance(feature_importance, save_path='results/figures/test_fi.png')
        print("   ✅ Feature importance plotted")
        
        plot_score_distributions(pos_scores1, neg_scores1, 'Adamic-Adar',
                                save_path='results/figures/test_dist.png')
        print("   ✅ Score distributions plotted")
        
        # Test comparison table
        print("\n3. Creating comparison table...")
        comparison_df, styled_df = create_comparison_table(all_results)
        print(comparison_df.to_string(index=False))
        print("   ✅ Comparison table created")
        
        # Generate comprehensive report
        print("\n4. Generating comprehensive report...")
        generate_link_prediction_report(
            all_results,
            'results/tables/link_prediction_report.html',
            roc_data=roc_data,
            pr_data=pr_data,
            confusion_matrices=confusion_matrices,
            feature_importance=feature_importance,
            score_distributions=score_distributions
        )
        print("   ✅ Comprehensive report generated")
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)
    
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"\n✗ Error: {e}")
        raise


if __name__ == "__main__":
    main()

