"""
Report generation module for network analysis results.

This module provides functions for generating comprehensive reports
in markdown and LaTeX formats.
"""

import os
import time
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_markdown_report(
    results_dict: Dict[str, Any],
    output_path: str,
    figures_dir: Optional[str] = None
) -> None:
    """
    Generate comprehensive markdown report.
    
    Args:
        results_dict: Dictionary containing all analysis results
        output_path: Path to save markdown report
        figures_dir: Directory containing figure files (for embedding)
    
    Raises:
        ValueError: If results_dict is empty
    """
    if not results_dict:
        raise ValueError("results_dict is empty")
    
    logger.info(f"Generating markdown report: {output_path}")
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    with open(output_path, 'w') as f:
        # Title and metadata
        f.write("# Amazon Product Co-Purchasing Network Analysis Report\n\n")
        f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        if 'summary' in results_dict:
            summary = results_dict['summary']
            f.write(f"{summary.get('description', 'Comprehensive analysis of Amazon product co-purchasing network.')}\n\n")
            f.write("**Key Findings:**\n\n")
            if 'key_findings' in summary:
                for finding in summary['key_findings']:
                    f.write(f"- {finding}\n")
            f.write("\n")
        else:
            f.write("This report presents a comprehensive analysis of the Amazon product co-purchasing network.\n\n")
        f.write("---\n\n")
        
        # Data Overview
        f.write("## 1. Data Overview\n\n")
        if 'data_overview' in results_dict:
            data = results_dict['data_overview']
            f.write("### Network Statistics\n\n")
            f.write(f"- **Number of Nodes**: {data.get('num_nodes', 'N/A'):,}\n")
            f.write(f"- **Number of Edges**: {data.get('num_edges', 'N/A'):,}\n")
            f.write(f"- **Density**: {data.get('density', 0):.6f}\n")
            f.write(f"- **Average Degree**: {data.get('avg_degree', 0):.4f}\n")
            if 'largest_component' in data:
                f.write(f"- **Largest Component**: {data['largest_component']:,} nodes\n")
            f.write("\n")
        else:
            f.write("Data overview information not available.\n\n")
        f.write("---\n\n")
        
        # Centrality Analysis Results
        f.write("## 2. Centrality Analysis Results\n\n")
        if 'centrality' in results_dict:
            centrality = results_dict['centrality']
            f.write("### Top Central Nodes\n\n")
            
            if 'top_nodes' in centrality:
                top_nodes = centrality['top_nodes']
                if isinstance(top_nodes, pd.DataFrame):
                    f.write(top_nodes.to_markdown(index=False))
                    f.write("\n\n")
                elif isinstance(top_nodes, dict):
                    for method, nodes in top_nodes.items():
                        f.write(f"**{method.replace('_', ' ').title()}:**\n\n")
                        if isinstance(nodes, pd.DataFrame):
                            f.write(nodes.head(10).to_markdown(index=False))
                            f.write("\n\n")
            
            if 'correlation_matrix' in centrality:
                f.write("### Centrality Correlation\n\n")
                corr_df = centrality['correlation_matrix']
                if isinstance(corr_df, pd.DataFrame):
                    f.write(corr_df.to_markdown())
                    f.write("\n\n")
            
            # Embed figure if available
            if figures_dir:
                centrality_fig = Path(figures_dir) / 'centrality_top_k_comparison.png'
                if centrality_fig.exists():
                    f.write(f"![Top Central Nodes]({centrality_fig.relative_to(Path(output_path).parent)})\n\n")
        else:
            f.write("Centrality analysis results not available.\n\n")
        f.write("---\n\n")
        
        # Community Detection Results
        f.write("## 3. Community Detection Results\n\n")
        if 'communities' in results_dict:
            communities = results_dict['communities']
            f.write("### Algorithm Comparison\n\n")
            
            if 'comparison' in communities:
                comp_df = communities['comparison']
                if isinstance(comp_df, pd.DataFrame):
                    f.write(comp_df.to_markdown(index=False))
                    f.write("\n\n")
            
            if 'best_method' in communities:
                f.write(f"**Best Performing Method**: {communities['best_method']}\n\n")
            
            if 'num_communities' in communities:
                f.write(f"**Number of Communities Detected**: {communities['num_communities']:,}\n\n")
            
            # Embed figure if available
            if figures_dir:
                comm_fig = Path(figures_dir) / 'community_size_distribution.png'
                if comm_fig.exists():
                    f.write(f"![Community Size Distribution]({comm_fig.relative_to(Path(output_path).parent)})\n\n")
        else:
            f.write("Community detection results not available.\n\n")
        f.write("---\n\n")
        
        # Link Prediction Results
        f.write("## 4. Link Prediction Results\n\n")
        if 'link_prediction' in results_dict:
            lp = results_dict['link_prediction']
            f.write("### Method Comparison\n\n")
            
            if 'comparison' in lp:
                comp_df = lp['comparison']
                if isinstance(comp_df, pd.DataFrame):
                    f.write(comp_df.to_markdown(index=False))
                    f.write("\n\n")
            
            if 'best_method' in lp:
                f.write(f"**Best Performing Method**: {lp['best_method']}\n\n")
                if 'best_metrics' in lp:
                    metrics = lp['best_metrics']
                    f.write(f"- **F1 Score**: {metrics.get('f1', 0):.4f}\n")
                    f.write(f"- **AUC-ROC**: {metrics.get('auc_roc', 0):.4f}\n")
                    f.write(f"- **AUC-PR**: {metrics.get('auc_pr', 0):.4f}\n\n")
            
            # Embed figure if available
            if figures_dir:
                lp_fig = Path(figures_dir) / 'link_prediction_roc_curves.png'
                if lp_fig.exists():
                    f.write(f"![ROC Curves]({lp_fig.relative_to(Path(output_path).parent)})\n\n")
        else:
            f.write("Link prediction results not available.\n\n")
        f.write("---\n\n")
        
        # Performance Analysis
        f.write("## 5. Performance Analysis\n\n")
        if 'performance' in results_dict:
            perf = results_dict['performance']
            f.write("### Execution Times\n\n")
            
            if 'timings' in perf:
                timings_df = perf['timings']
                if isinstance(timings_df, pd.DataFrame):
                    f.write(timings_df.to_markdown(index=False))
                    f.write("\n\n")
            
            if 'total_time' in perf:
                f.write(f"**Total Execution Time**: {perf['total_time']:.2f} seconds ({perf['total_time']/60:.2f} minutes)\n\n")
        else:
            f.write("Performance analysis results not available.\n\n")
        f.write("---\n\n")
        
        # Conclusions
        f.write("## 6. Conclusions\n\n")
        if 'conclusions' in results_dict:
            conclusions = results_dict['conclusions']
            if isinstance(conclusions, list):
                for conclusion in conclusions:
                    f.write(f"- {conclusion}\n")
            else:
                f.write(f"{conclusions}\n")
            f.write("\n")
        else:
            f.write("### Key Insights\n\n")
            f.write("1. The network exhibits scale-free properties with power-law degree distribution.\n")
            f.write("2. Community detection reveals meaningful product clusters.\n")
            f.write("3. Link prediction methods show varying performance, with ML-based approaches performing best.\n")
            f.write("4. Centrality analysis identifies key hub products in the network.\n\n")
        
        # References
        f.write("---\n\n")
        f.write("## References\n\n")
        f.write("- Amazon Product Co-Purchasing Network Dataset: SNAP Stanford\n")
        f.write("- NetworkX Documentation: https://networkx.org/\n")
        f.write("- Scikit-learn Documentation: https://scikit-learn.org/\n\n")
    
    logger.info(f"Markdown report saved to {output_path}")


def create_latex_tables(
    dataframes_dict: Dict[str, pd.DataFrame],
    output_dir: str
) -> None:
    """
    Convert pandas DataFrames to LaTeX tables.
    
    Args:
        dataframes_dict: Dictionary mapping table names to DataFrames
        output_dir: Directory to save LaTeX tables
    
    Raises:
        ValueError: If dataframes_dict is empty
    """
    if not dataframes_dict:
        raise ValueError("dataframes_dict is empty")
    
    logger.info(f"Creating LaTeX tables in {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    for table_name, df in dataframes_dict.items():
        if df.empty:
            logger.warning(f"DataFrame {table_name} is empty, skipping")
            continue
        
        # Clean table name for filename
        safe_name = table_name.replace(' ', '_').replace('/', '_').lower()
        output_path = os.path.join(output_dir, f"{safe_name}.tex")
        
        # Convert to LaTeX
        latex_str = df.to_latex(
            index=False,
            float_format="%.4f",
            escape=False,
            caption=f"{table_name.replace('_', ' ').title()}",
            label=f"tab:{safe_name}"
        )
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write(latex_str)
        
        logger.info(f"LaTeX table saved: {output_path}")
    
    logger.info(f"All LaTeX tables created in {output_dir}")


def compile_results_summary(
    all_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create summary of key findings from all results.
    
    Args:
        all_results: Dictionary containing all analysis results
    
    Returns:
        Dictionary with:
            - top_products: List of top products/nodes
            - best_algorithms: Best performing algorithms for each task
            - key_metrics: Key performance metrics
            - insights: List of insights
    """
    logger.info("Compiling results summary...")
    
    summary = {
        'top_products': [],
        'best_algorithms': {},
        'key_metrics': {},
        'insights': []
    }
    
    # Extract top products from centrality
    if 'centrality' in all_results and 'top_nodes' in all_results['centrality']:
        top_nodes = all_results['centrality']['top_nodes']
        if isinstance(top_nodes, dict):
            # Get top nodes from first method
            first_method = list(top_nodes.keys())[0]
            if isinstance(top_nodes[first_method], pd.DataFrame):
                top_products = top_nodes[first_method].head(10)['node_id'].tolist()
                summary['top_products'] = top_products
    
    # Extract best algorithms
    if 'communities' in all_results and 'best_method' in all_results['communities']:
        summary['best_algorithms']['community_detection'] = all_results['communities']['best_method']
    
    if 'link_prediction' in all_results and 'best_method' in all_results['link_prediction']:
        summary['best_algorithms']['link_prediction'] = all_results['link_prediction']['best_method']
    
    # Extract key metrics
    if 'link_prediction' in all_results and 'best_metrics' in all_results['link_prediction']:
        summary['key_metrics']['link_prediction'] = all_results['link_prediction']['best_metrics']
    
    if 'communities' in all_results and 'best_metrics' in all_results['communities']:
        summary['key_metrics']['community_detection'] = all_results['communities']['best_metrics']
    
    # Generate insights
    insights = []
    
    if 'data_overview' in all_results:
        data = all_results['data_overview']
        if data.get('num_nodes', 0) > 100000:
            insights.append("Large-scale network with over 100,000 nodes")
        if data.get('density', 0) < 0.01:
            insights.append("Sparse network with low density")
    
    if 'communities' in all_results:
        insights.append("Community detection successfully identified product clusters")
    
    if 'link_prediction' in all_results:
        best_f1 = all_results['link_prediction'].get('best_metrics', {}).get('f1', 0)
        if best_f1 > 0.8:
            insights.append(f"Link prediction achieved high accuracy (F1={best_f1:.2f})")
    
    summary['insights'] = insights
    
    logger.info("Results summary compiled")
    
    return summary


def main():
    """
    Main function demonstrating usage of the report generator module.
    """
    print("=" * 60)
    print("Report Generator Module - Demo")
    print("=" * 60)
    
    try:
        # Create sample results
        sample_results = {
            'summary': {
                'description': 'Sample analysis report',
                'key_findings': [
                    'Network has scale-free properties',
                    'Community detection identified meaningful clusters',
                    'ML-based link prediction performs best'
                ]
            },
            'data_overview': {
                'num_nodes': 334863,
                'num_edges': 925872,
                'density': 0.000016,
                'avg_degree': 5.53,
                'largest_component': 334863
            },
            'centrality': {
                'top_nodes': {
                    'pagerank': pd.DataFrame({
                        'node_id': [1, 2, 3, 4, 5],
                        'score': [0.001, 0.0009, 0.0008, 0.0007, 0.0006]
                    })
                }
            },
            'communities': {
                'best_method': 'Louvain',
                'num_communities': 75149
            },
            'link_prediction': {
                'best_method': 'Random Forest',
                'best_metrics': {
                    'f1': 0.89,
                    'auc_roc': 0.95,
                    'auc_pr': 0.92
                }
            },
            'performance': {
                'total_time': 3600
            }
        }
        
        # Generate markdown report
        print("\n1. Generating markdown report...")
        generate_markdown_report(sample_results, 'results/tables/sample_report.md')
        print("   ✅ Markdown report generated")
        
        # Create LaTeX tables
        print("\n2. Creating LaTeX tables...")
        tables = {
            'centrality_comparison': pd.DataFrame({
                'Method': ['PageRank', 'Degree', 'Betweenness'],
                'Top Node': [1, 2, 3],
                'Score': [0.001, 0.002, 0.0015]
            })
        }
        create_latex_tables(tables, 'results/tables/latex')
        print("   ✅ LaTeX tables created")
        
        # Compile summary
        print("\n3. Compiling results summary...")
        summary = compile_results_summary(sample_results)
        print(f"   ✅ Summary compiled: {len(summary['insights'])} insights")
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)
    
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"\n✗ Error: {e}")
        raise


if __name__ == "__main__":
    main()

