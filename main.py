#!/usr/bin/env python3
"""
Master execution script for Amazon Network Analysis pipeline.

This script orchestrates the entire analysis pipeline including data loading,
preprocessing, centrality analysis, community detection, link prediction,
and report generation.
"""

import os
import sys
import argparse
import logging
import time
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import project modules
from data_loader import download_dataset, load_graph, load_saved_graph, save_graph
from preprocessing import (
    remove_self_loops,
    get_largest_component,
    create_train_test_split,
    save_splits
)
from centrality_analysis import compare_centrality_measures
from community_detection import run_all_community_detection
from link_prediction import evaluate_link_prediction
from ml_link_prediction import (
    prepare_training_data,
    train_random_forest,
    evaluate_ml_model
)
from network_properties import generate_network_report
from visualization import create_all_visualizations, export_to_gephi
from report_generator import generate_markdown_report, compile_results_summary
from parallel_processing import parallel_pagerank, parallel_betweenness
import pandas as pd

# Try to import rich for beautiful progress bars
try:
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.console import Console
    from rich.logging import RichHandler
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    from tqdm import tqdm

# Configure logging
def setup_logging(log_file: str, log_level: str = 'INFO'):
    """Setup logging to both console and file."""
    log_level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR
    }
    
    level = log_level_map.get(log_level.upper(), logging.INFO)
    
    # Create logs directory
    os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else '.', exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            RichHandler() if RICH_AVAILABLE else logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def save_checkpoint(checkpoint_data: Dict, checkpoint_path: str):
    """Save checkpoint data."""
    os.makedirs(os.path.dirname(checkpoint_path) if os.path.dirname(checkpoint_path) else '.', exist_ok=True)
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    logger.info(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(checkpoint_path: str) -> Optional[Dict]:
    """Load checkpoint data."""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            return pickle.load(f)
    return None


def run_full_pipeline(
    data_dir: str,
    output_dir: str,
    skip_download: bool = False,
    modules: List[str] = ['all'],
    parallel: bool = False,
    n_jobs: int = -1,
    checkpoint_dir: Optional[str] = None
) -> Dict:
    """
    Run the full analysis pipeline.
    
    Args:
        data_dir: Path to data directory
        output_dir: Path for results
        skip_download: Whether to skip data download
        modules: List of modules to run
        parallel: Whether to use parallel processing
        n_jobs: Number of parallel jobs
        checkpoint_dir: Directory for checkpoints
    
    Returns:
        Dictionary with all results
    """
    logger.info("=" * 70)
    logger.info("Starting Full Analysis Pipeline")
    logger.info("=" * 70)
    
    pipeline_start = time.time()
    all_results = {}
    checkpoint_data = {}
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/figures", exist_ok=True)
    os.makedirs(f"{output_dir}/tables", exist_ok=True)
    os.makedirs(f"{output_dir}/logs", exist_ok=True)
    
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Step 1: Load/Download and Preprocess Data
    if 'all' in modules or 'data' in modules:
        logger.info("\n" + "=" * 70)
        logger.info("STEP 1: Data Loading and Preprocessing")
        logger.info("=" * 70)
        
        try:
            step_start = time.time()
            
            # Download data if needed
            if not skip_download:
                logger.info("Downloading dataset...")
                download_dataset(data_dir)
            
            # Load graph
            logger.info("Loading graph...")
            graph_path = f"{data_dir}/raw/com-amazon.ungraph.txt.gz"
            if os.path.exists(graph_path):
                G = load_graph(graph_path)
                logger.info(f"Graph loaded: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
            else:
                logger.warning(f"Graph file not found: {graph_path}")
                G = None
            
            if G is not None:
                # Preprocessing
                logger.info("Preprocessing graph...")
                G = remove_self_loops(G)
                G = get_largest_component(G)
                logger.info(f"Cleaned graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
                
                # Save cleaned graph
                cleaned_path = f"{data_dir}/processed/amazon_graph_cleaned.pkl"
                os.makedirs(os.path.dirname(cleaned_path), exist_ok=True)
                save_graph(G, cleaned_path)
                
                # Create train/test split
                logger.info("Creating train/test split...")
                G_train, pos_test, neg_test = create_train_test_split(G, test_ratio=0.2, seed=42)
                save_splits(G_train, pos_test, neg_test, f"{data_dir}/processed/splits")
                
                all_results['data_overview'] = {
                    'num_nodes': G.number_of_nodes(),
                    'num_edges': G.number_of_edges(),
                    'density': 2 * G.number_of_edges() / (G.number_of_nodes() * (G.number_of_nodes() - 1)),
                    'avg_degree': 2 * G.number_of_edges() / G.number_of_nodes(),
                    'largest_component': G.number_of_nodes()
                }
                
                checkpoint_data['graph'] = G
                checkpoint_data['graph_cleaned'] = True
                
                step_time = time.time() - step_start
                logger.info(f"Step 1 completed in {step_time:.2f} seconds")
            else:
                logger.error("Failed to load graph")
        
        except Exception as e:
            logger.error(f"Step 1 failed: {e}", exc_info=True)
            if checkpoint_dir:
                save_checkpoint(checkpoint_data, f"{checkpoint_dir}/step1_checkpoint.pkl")
    
    # Step 2: Centrality Analysis
    if 'all' in modules or 'centrality' in modules:
        logger.info("\n" + "=" * 70)
        logger.info("STEP 2: Centrality Analysis")
        logger.info("=" * 70)
        
        try:
            step_start = time.time()
            
            # Load graph
            graph_path = f"{data_dir}/processed/amazon_graph_cleaned.pkl"
            if not os.path.exists(graph_path):
                logger.warning(f"Cleaned graph not found: {graph_path}, skipping centrality analysis")
            else:
                G = load_saved_graph(graph_path)
                
                if parallel:
                    logger.info(f"Running parallel centrality analysis (n_jobs={n_jobs})...")
                    pr_result = parallel_pagerank(G, n_jobs=n_jobs)
                    bt_result = parallel_betweenness(G, n_jobs=n_jobs, k=1000)
                    all_results['centrality'] = {
                        'pagerank': pr_result,
                        'betweenness': bt_result
                    }
                else:
                    logger.info("Running centrality analysis...")
                    centrality_results = compare_centrality_measures(G, k=100)
                    all_results['centrality'] = centrality_results
                
                checkpoint_data['centrality'] = True
                step_time = time.time() - step_start
                logger.info(f"Step 2 completed in {step_time:.2f} seconds")
        
        except Exception as e:
            logger.error(f"Step 2 failed: {e}", exc_info=True)
            if checkpoint_dir:
                save_checkpoint(checkpoint_data, f"{checkpoint_dir}/step2_checkpoint.pkl")
    
    # Step 3: Community Detection
    if 'all' in modules or 'communities' in modules:
        logger.info("\n" + "=" * 70)
        logger.info("STEP 3: Community Detection")
        logger.info("=" * 70)
        
        try:
            step_start = time.time()
            
            graph_path = f"{data_dir}/processed/amazon_graph_cleaned.pkl"
            if not os.path.exists(graph_path):
                logger.warning(f"Cleaned graph not found: {graph_path}, skipping community detection")
            else:
                G = load_saved_graph(graph_path)
                
                logger.info("Running community detection algorithms...")
                community_results = run_all_community_detection(G, louvain_resolution=1.0, seed=42)
                
                # Find best method
                best_method = None
                best_modularity = -1
                for method in ['louvain', 'label_propagation', 'greedy_modularity']:
                    if method in community_results and community_results[method] is not None:
                        _, mod = community_results[method]
                        if mod > best_modularity:
                            best_modularity = mod
                            best_method = method
                
                all_results['communities'] = {
                    'results': community_results,
                    'best_method': best_method,
                    'best_modularity': best_modularity
                }
                
                checkpoint_data['communities'] = True
                step_time = time.time() - step_start
                logger.info(f"Step 3 completed in {step_time:.2f} seconds")
        
        except Exception as e:
            logger.error(f"Step 3 failed: {e}", exc_info=True)
            if checkpoint_dir:
                save_checkpoint(checkpoint_data, f"{checkpoint_dir}/step3_checkpoint.pkl")
    
    # Step 4: Link Prediction
    if 'all' in modules or 'link_prediction' in modules:
        logger.info("\n" + "=" * 70)
        logger.info("STEP 4: Link Prediction")
        logger.info("=" * 70)
        
        try:
            step_start = time.time()
            
            # Load train graph and test edges
            train_path = f"{data_dir}/processed/splits/train_graph.pkl"
            pos_test_path = f"{data_dir}/processed/splits/positive_test_edges.pkl"
            neg_test_path = f"{data_dir}/processed/splits/negative_test_edges.pkl"
            
            if not all(os.path.exists(p) for p in [train_path, pos_test_path, neg_test_path]):
                logger.warning("Train/test splits not found, skipping link prediction")
            else:
                with open(train_path, 'rb') as f:
                    G_train = pickle.load(f)
                with open(pos_test_path, 'rb') as f:
                    pos_test = pickle.load(f)
                with open(neg_test_path, 'rb') as f:
                    neg_test = pickle.load(f)
                
                # Use sample for faster computation
                sample_size = min(1000, len(pos_test))
                pos_sample = pos_test[:sample_size]
                neg_sample = neg_test[:sample_size]
                
                # Evaluate similarity methods
                logger.info("Evaluating similarity-based methods...")
                similarity_results = {}
                for method in ['common_neighbors', 'adamic_adar', 'jaccard']:
                    try:
                        metrics = evaluate_link_prediction(G_train, pos_sample, neg_sample, method=method)
                        similarity_results[method] = metrics
                    except Exception as e:
                        logger.warning(f"Failed to evaluate {method}: {e}")
                
                # Train and evaluate ML model
                logger.info("Training ML model...")
                try:
                    train_size = min(2000, len(pos_test))
                    X_train, y_train = prepare_training_data(
                        G_train, pos_test[:train_size], neg_test[:train_size]
                    )
                    model, scaler = train_random_forest(X_train, y_train, n_estimators=100, random_state=42)
                    ml_metrics, _ = evaluate_ml_model(
                        model, G_train, pos_sample, neg_sample, scaler=scaler
                    )
                    similarity_results['random_forest'] = ml_metrics
                except Exception as e:
                    logger.warning(f"ML model training failed: {e}")
                
                # Find best method
                best_method = None
                best_f1 = -1
                for method, metrics in similarity_results.items():
                    if isinstance(metrics, dict) and metrics.get('f1', 0) > best_f1:
                        best_f1 = metrics['f1']
                        best_method = method
                
                all_results['link_prediction'] = {
                    'results': similarity_results,
                    'best_method': best_method,
                    'best_metrics': similarity_results.get(best_method, {}) if best_method else {}
                }
                
                checkpoint_data['link_prediction'] = True
                step_time = time.time() - step_start
                logger.info(f"Step 4 completed in {step_time:.2f} seconds")
        
        except Exception as e:
            logger.error(f"Step 4 failed: {e}", exc_info=True)
            if checkpoint_dir:
                save_checkpoint(checkpoint_data, f"{checkpoint_dir}/step4_checkpoint.pkl")
    
    # Step 5: Network Properties Analysis
    if 'all' in modules or 'properties' in modules:
        logger.info("\n" + "=" * 70)
        logger.info("STEP 5: Network Properties Analysis")
        logger.info("=" * 70)
        
        try:
            step_start = time.time()
            
            graph_path = f"{data_dir}/processed/amazon_graph_cleaned.pkl"
            if not os.path.exists(graph_path):
                logger.warning(f"Cleaned graph not found: {graph_path}, skipping properties analysis")
            else:
                G = load_saved_graph(graph_path)
                
                logger.info("Computing network properties...")
                properties_report = generate_network_report(
                    G,
                    f"{output_dir}/tables/network_properties_report.md",
                    sample_size=1000
                )
                all_results['network_properties'] = properties_report
                
                checkpoint_data['properties'] = True
                step_time = time.time() - step_start
                logger.info(f"Step 5 completed in {step_time:.2f} seconds")
        
        except Exception as e:
            logger.error(f"Step 5 failed: {e}", exc_info=True)
            if checkpoint_dir:
                save_checkpoint(checkpoint_data, f"{checkpoint_dir}/step5_checkpoint.pkl")
    
    # Step 6: Generate Visualizations
    if 'all' in modules or 'visualization' in modules:
        logger.info("\n" + "=" * 70)
        logger.info("STEP 6: Generate Visualizations")
        logger.info("=" * 70)
        
        try:
            step_start = time.time()
            
            logger.info("Creating all visualizations...")
            create_all_visualizations(
                f"{data_dir}/processed",
                f"{output_dir}/figures"
            )
            
            checkpoint_data['visualizations'] = True
            step_time = time.time() - step_start
            logger.info(f"Step 6 completed in {step_time:.2f} seconds")
        
        except Exception as e:
            logger.error(f"Step 6 failed: {e}", exc_info=True)
            if checkpoint_dir:
                save_checkpoint(checkpoint_data, f"{checkpoint_dir}/step6_checkpoint.pkl")
    
    # Step 7: Create Final Report
    logger.info("\n" + "=" * 70)
    logger.info("STEP 7: Generate Final Report")
    logger.info("=" * 70)
    
    try:
        step_start = time.time()
        
        # Compile summary
        summary = compile_results_summary(all_results)
        all_results['summary'] = summary
        
        # Generate markdown report
        logger.info("Generating markdown report...")
        generate_markdown_report(
            all_results,
            f"{output_dir}/tables/final_report.md",
            figures_dir=f"{output_dir}/figures"
        )
        
        # Save results as JSON
        results_json_path = f"{output_dir}/all_results.json"
        with open(results_json_path, 'w') as f:
            # Convert DataFrames to dict for JSON serialization
            json_results = {}
            for key, value in all_results.items():
                if isinstance(value, dict):
                    json_results[key] = {
                        k: v.to_dict() if isinstance(v, pd.DataFrame) else v
                        for k, v in value.items()
                    }
                else:
                    json_results[key] = value
            json.dump(json_results, f, indent=2, default=str)
        
        step_time = time.time() - step_start
        logger.info(f"Step 7 completed in {step_time:.2f} seconds")
    
    except Exception as e:
        logger.error(f"Step 7 failed: {e}", exc_info=True)
    
    # Final summary
    total_time = time.time() - pipeline_start
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETED")
    logger.info("=" * 70)
    logger.info(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"  - Figures: {output_dir}/figures/")
    logger.info(f"  - Tables: {output_dir}/tables/")
    logger.info(f"  - Logs: {output_dir}/logs/")
    
    return all_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Amazon Network Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --data-dir ./data --output-dir ./results --modules all
  python main.py --data-dir ./data --output-dir ./results --modules centrality,communities --parallel --n-jobs 4
  python main.py --data-dir ./data --output-dir ./results --modules link_prediction --skip-download
        """
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data',
        help='Path to data directory (default: ./data)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results',
        help='Path for results output (default: ./results)'
    )
    
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip data download step'
    )
    
    parser.add_argument(
        '--modules',
        type=str,
        default='all',
        help='Comma-separated list of modules to run: all, data, centrality, communities, link_prediction, properties, visualization (default: all)'
    )
    
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Use parallel processing'
    )
    
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=-1,
        help='Number of parallel jobs (-1 for all cores, default: -1)'
    )
    
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default=None,
        help='Directory for checkpoints (default: None)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = f"{args.output_dir}/logs/pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logging(log_file, args.log_level)
    
    # Parse modules
    modules = [m.strip() for m in args.modules.split(',')]
    
    # Run pipeline
    try:
        results = run_full_pipeline(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            skip_download=args.skip_download,
            modules=modules,
            parallel=args.parallel,
            n_jobs=args.n_jobs,
            checkpoint_dir=args.checkpoint_dir or f"{args.output_dir}/checkpoints"
        )
        
        logger.info("\n✅ Pipeline completed successfully!")
        return 0
    
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Pipeline interrupted by user")
        return 1
    
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

