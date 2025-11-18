"""
Comprehensive spam analysis pipeline.

This module integrates all spam-related analysis modules to provide
a complete experimental framework for studying spam farms, detection methods,
and their effectiveness.

Pipeline capabilities:
1. Complete spam experiments with all detection methods
2. Parameter sweeps for sensitivity analysis
3. Strategy comparison (different spam types)
4. Comprehensive reporting
5. Demo notebook generation

Reference: Mining of Massive Datasets, Chapter 5 (Link Analysis)
"""

import os
import sys
import time
import logging
import json
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Set, Tuple, Optional, Union, Any
from datetime import datetime
from dataclasses import dataclass, asdict
from tqdm import tqdm
from itertools import product
import networkx as nx

# Try to import joblib for parallel processing
try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
    logger = logging.getLogger(__name__)
    logger.warning("joblib not available, parallel processing disabled")

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


@dataclass
class ExperimentResults:
    """Container for experiment results."""
    experiment_id: str
    config: Dict
    G_original: Optional[Any] = None  # NetworkX graph (not serialized)
    G_with_spam: Optional[Any] = None  # NetworkX graph (not serialized)
    spam_metadata: Dict = None
    baseline_pagerank: Dict = None
    baseline_hits: Dict = None
    infected_pagerank: Dict = None
    trustrank_scores: Dict = None
    spam_mass: Dict = None
    structural_detections: Dict = None
    detection_evaluations: Dict = None
    effectiveness_analysis: Dict = None
    network_damage: Dict = None
    execution_time: float = 0.0
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.spam_metadata is None:
            self.spam_metadata = {}
        if self.baseline_pagerank is None:
            self.baseline_pagerank = {}
        if self.baseline_hits is None:
            self.baseline_hits = {}
        if self.infected_pagerank is None:
            self.infected_pagerank = {}
        if self.trustrank_scores is None:
            self.trustrank_scores = {}
        if self.spam_mass is None:
            self.spam_mass = {}
        if self.structural_detections is None:
            self.structural_detections = {}
        if self.detection_evaluations is None:
            self.detection_evaluations = {}
        if self.effectiveness_analysis is None:
            self.effectiveness_analysis = {}
        if self.network_damage is None:
            self.network_damage = {}
    
    def to_dict(self):
        """Convert to dictionary (excluding graph objects)."""
        result = asdict(self)
        result['G_original'] = None
        result['G_with_spam'] = None
        return result
    
    def save(self, filepath: str):
        """Save results to file (excluding graph objects)."""
        data = self.to_dict()
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    @classmethod
    def load(cls, filepath: str):
        """Load results from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)


def run_complete_spam_experiment(
    G_original: nx.Graph,
    experiment_config: Dict,
    save_intermediate: bool = True,
    output_dir: Optional[str] = None
) -> ExperimentResults:
    """
    Run complete spam experiment with all detection methods.
    
    Pipeline steps:
    1. Load original graph
    2. Compute baseline PageRank/HITS
    3. Create spam farm(s)
    4. Compute PageRank on infected graph
    5. Run TrustRank
    6. Calculate spam mass
    7. Run structural detection
    8. Evaluate all methods
    9. Analyze effectiveness
    10. Generate comprehensive report
    
    Args:
        G_original: Original graph without spam
        experiment_config: Dictionary with experiment parameters:
            - spam_type: 'simple', 'multiple', 'collaborative'
            - m: number of supporting pages
            - external_links: number of external links
            - beta: taxation parameter (default: 0.85)
            - detection_methods: list of methods ['trustrank', 'spam_mass', 'structural']
            - trusted_nodes: set of trusted nodes for TrustRank
            - target_node: optional target node (default: random)
        save_intermediate: Whether to save intermediate results
        output_dir: Directory to save results
    
    Returns:
        ExperimentResults object with all data
    """
    start_time = time.time()
    experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    logger.info("=" * 60)
    logger.info(f"Starting spam experiment: {experiment_id}")
    logger.info("=" * 60)
    
    # Initialize results
    results = ExperimentResults(
        experiment_id=experiment_id,
        config=experiment_config
    )
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Step 1: Compute baseline metrics
        logger.info("Step 1: Computing baseline PageRank and HITS...")
        beta = experiment_config.get('beta', 0.85)
        
        baseline_pr = nx.pagerank(G_original, alpha=beta, max_iter=100)
        results.baseline_pagerank = baseline_pr
        
        try:
            hubs, authorities = nx.hits(G_original, max_iter=100)
            results.baseline_hits = {'hubs': hubs, 'authorities': authorities}
        except Exception as e:
            logger.warning(f"Error computing HITS: {e}")
            results.baseline_hits = {}
        
        # Step 2: Create spam farm
        logger.info("Step 2: Creating spam farm...")
        spam_type = experiment_config.get('spam_type', 'simple')
        m = experiment_config.get('m', 1000)
        external_links = experiment_config.get('external_links', 10)
        target_node = experiment_config.get('target_node', None)
        
        G_with_spam, spam_metadata = _create_spam_farm(
            G_original.copy(), spam_type, m, external_links, target_node
        )
        results.G_with_spam = G_with_spam
        results.spam_metadata = spam_metadata
        
        # Extract target nodes
        target_nodes = _extract_target_nodes(spam_metadata)
        true_spam = _extract_all_spam_nodes(spam_metadata)
        
        logger.info(f"  Spam farm created: {len(true_spam)} spam nodes, {len(target_nodes)} targets")
        
        # Step 3: Compute PageRank on infected graph
        logger.info("Step 3: Computing PageRank on infected graph...")
        infected_pr = nx.pagerank(G_with_spam, alpha=beta, max_iter=100)
        results.infected_pagerank = infected_pr
        
        # Step 4: Run TrustRank (if enabled)
        detection_methods = experiment_config.get('detection_methods', ['trustrank', 'spam_mass', 'structural'])
        
        if 'trustrank' in detection_methods:
            logger.info("Step 4: Running TrustRank...")
            try:
                from trustrank import compute_trustrank, select_trusted_pages
                
                trusted_nodes = experiment_config.get('trusted_nodes', None)
                if trusted_nodes is None:
                    # Select top-k by PageRank from original graph
                    trusted_nodes = select_trusted_pages(
                        G_original, method='top_pagerank', k=100, pagerank_scores=baseline_pr
                    )
                
                trustrank_scores, _ = compute_trustrank(
                    G_with_spam, trusted_nodes, beta=beta, max_iter=100
                )
                results.trustrank_scores = trustrank_scores
                logger.info(f"  TrustRank computed for {len(trustrank_scores)} nodes")
            except Exception as e:
                logger.error(f"Error running TrustRank: {e}", exc_info=True)
                results.trustrank_scores = {}
        
        # Step 5: Calculate spam mass (if enabled)
        if 'spam_mass' in detection_methods:
            logger.info("Step 5: Calculating spam mass...")
            try:
                from spam_mass import calculate_spam_mass
                
                if results.trustrank_scores:
                    spam_mass = calculate_spam_mass(infected_pr, results.trustrank_scores)
                    results.spam_mass = spam_mass
                    logger.info(f"  Spam mass calculated for {len(spam_mass)} nodes")
                else:
                    logger.warning("TrustRank not available, skipping spam mass")
                    results.spam_mass = {}
            except Exception as e:
                logger.error(f"Error calculating spam mass: {e}", exc_info=True)
                results.spam_mass = {}
        
        # Step 6: Run structural detection (if enabled)
        if 'structural' in detection_methods:
            logger.info("Step 6: Running structural detection...")
            try:
                from structural_spam_detection import ensemble_structural_detection
                
                structural_results = ensemble_structural_detection(
                    G_with_spam, methods=['reciprocal', 'star', 'honeypot', 'clustering']
                )
                
                # Convert to set of detected nodes
                detected_structural = set(structural_results['node_id'].tolist())
                results.structural_detections = {
                    'detected_nodes': list(detected_structural),
                    'results_df': structural_results.to_dict('records')
                }
                logger.info(f"  Structural detection found {len(detected_structural)} suspicious nodes")
            except Exception as e:
                logger.error(f"Error running structural detection: {e}", exc_info=True)
                results.structural_detections = {}
        
        # Step 7: Evaluate detection methods
        logger.info("Step 7: Evaluating detection methods...")
        try:
            from spam_detection_evaluation import (
                evaluate_detection_method,
                compare_detection_methods
            )
            
            # Collect all detections
            detection_results = {}
            all_nodes = set(G_with_spam.nodes())
            
            if results.trustrank_scores:
                # Use spam mass threshold for TrustRank-based detection
                if results.spam_mass:
                    threshold = 0.7
                    detected_trustrank = {
                        node for node, mass in results.spam_mass.items() 
                        if mass > threshold
                    }
                    detection_results['spam_mass'] = detected_trustrank
            
            if results.structural_detections:
                detected_structural = set(results.structural_detections.get('detected_nodes', []))
                detection_results['structural'] = detected_structural
            
            # Evaluate each method
            evaluations = {}
            for method_name, detected in detection_results.items():
                metrics, cm = evaluate_detection_method(detected, true_spam, all_nodes)
                evaluations[method_name] = {
                    'metrics': metrics,
                    'confusion_matrix': cm.tolist()
                }
            
            # Compare methods
            if len(detection_results) > 1:
                comparison_df, rankings = compare_detection_methods(true_spam, detection_results, all_nodes)
                evaluations['comparison'] = comparison_df.to_dict('records')
                evaluations['rankings'] = rankings.to_dict('records')
            
            results.detection_evaluations = evaluations
            logger.info(f"  Evaluated {len(evaluations)} detection methods")
        
        except Exception as e:
            logger.error(f"Error evaluating methods: {e}", exc_info=True)
            results.detection_evaluations = {}
        
        # Step 8: Analyze effectiveness
        logger.info("Step 8: Analyzing spam effectiveness...")
        try:
            from spam_effectiveness_analysis import (
                calculate_pagerank_amplification,
                analyze_spam_farm_roi,
                analyze_network_damage
            )
            
            # Amplification analysis
            amplification_df = calculate_pagerank_amplification(
                G_original, G_with_spam, target_nodes, spam_metadata, beta=beta
            )
            
            # ROI analysis
            boosts = {target: infected_pr.get(target, 0) - baseline_pr.get(target, 0) 
                     for target in target_nodes}
            roi_df = analyze_spam_farm_roi(spam_metadata, boosts)
            
            # Network damage
            network_damage = analyze_network_damage(G_original, G_with_spam)
            
            results.effectiveness_analysis = {
                'amplification': amplification_df.to_dict('records'),
                'roi': roi_df.to_dict('records'),
                'network_damage': network_damage
            }
            results.network_damage = network_damage
            
            logger.info("  Effectiveness analysis completed")
        
        except Exception as e:
            logger.error(f"Error analyzing effectiveness: {e}", exc_info=True)
            results.effectiveness_analysis = {}
        
        # Step 9: Save results
        if save_intermediate and output_dir:
            results_path = os.path.join(output_dir, f"{experiment_id}_results.json")
            results.save(results_path)
            logger.info(f"  Results saved to {results_path}")
        
        execution_time = time.time() - start_time
        results.execution_time = execution_time
        
        logger.info("=" * 60)
        logger.info(f"Experiment completed in {execution_time:.2f} seconds")
        logger.info("=" * 60)
        
        return results
    
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        results.execution_time = time.time() - start_time
        raise


def _create_spam_farm(
    G: nx.Graph,
    spam_type: str,
    m: int,
    external_links: int,
    target_node: Optional[Union[int, str]] = None
) -> Tuple[nx.Graph, Dict]:
    """Helper function to create spam farm based on type."""
    try:
        from spam_farm_generator import (
            create_simple_spam_farm,
            create_multiple_spam_farms,
            create_collaborative_spam_farms
        )
    except ImportError:
        logger.error("Cannot import spam_farm_generator")
        raise
    
    if target_node is None:
        target_node = list(G.nodes())[0]
    
    spam_metadata = {}
    
    if spam_type == 'simple':
        G_spam, spam_nodes, target = create_simple_spam_farm(
            G, target_node, m=m, external_links=external_links
        )
        spam_metadata[target] = {
            'm': m,
            'spam_nodes': spam_nodes,
            'farm_type': 'simple',
            'external_links': external_links
        }
    
    elif spam_type == 'multiple':
        num_farms = 1  # Default
        m_per_farm = m
        G_spam, farms_dict = create_multiple_spam_farms(
            G, num_farms=num_farms, m_per_farm=m_per_farm, external_links=external_links
        )
        for farm_id, (spam_nodes, target) in farms_dict.items():
            spam_metadata[target] = {
                'm': len(spam_nodes),
                'spam_nodes': spam_nodes,
                'farm_type': 'multiple',
                'external_links': external_links
            }
    
    elif spam_type == 'collaborative':
        num_farms = 2  # Default
        m_per_farm = m // num_farms
        G_spam, farms_dict, targets = create_collaborative_spam_farms(
            G, num_farms=num_farms, m_per_farm=m_per_farm, external_links=external_links
        )
        for target in targets:
            spam_metadata[target] = {
                'm': m_per_farm,
                'spam_nodes': [],  # Will be filled from farms_dict
                'farm_type': 'collaborative',
                'external_links': external_links
            }
    
    else:
        raise ValueError(f"Unknown spam type: {spam_type}")
    
    return G_spam, spam_metadata


def _extract_target_nodes(spam_metadata: Dict) -> List:
    """Extract all target nodes from spam metadata."""
    return list(spam_metadata.keys())


def _extract_all_spam_nodes(spam_metadata: Dict) -> Set:
    """Extract all spam nodes (targets + supporting pages) from metadata."""
    spam_nodes = set()
    for target, info in spam_metadata.items():
        spam_nodes.add(target)
        spam_nodes.update(info.get('spam_nodes', []))
    return spam_nodes


def run_parameter_sweep(
    G_original: nx.Graph,
    param_grid: Dict[str, List],
    output_dir: str,
    n_jobs: int = 1
) -> pd.DataFrame:
    """
    Run parameter sweep across all combinations.
    
    Args:
        G_original: Original graph
        param_grid: Dictionary mapping parameter names to lists of values
            Example: {'m': [100, 500, 1000], 'spam_type': ['simple', 'multiple']}
        output_dir: Directory to save results
        n_jobs: Number of parallel jobs (default: 1, -1 for all cores)
    
    Returns:
        Summary DataFrame with results for all combinations
    """
    logger.info("=" * 60)
    logger.info("Starting parameter sweep")
    logger.info("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))
    
    logger.info(f"Testing {len(combinations)} parameter combinations")
    
    # Default config
    default_config = {
        'beta': 0.85,
        'external_links': 10,
        'detection_methods': ['trustrank', 'spam_mass', 'structural'],
        'trusted_nodes': None
    }
    
    def run_single_experiment(combo):
        """Run experiment for single parameter combination."""
        config = default_config.copy()
        for param_name, param_value in zip(param_names, combo):
            config[param_name] = param_value
        
        try:
            results = run_complete_spam_experiment(
                G_original, config, save_intermediate=True, output_dir=output_dir
            )
            
            # Extract summary metrics
            summary = {
                **{name: val for name, val in zip(param_names, combo)},
                'experiment_id': results.experiment_id,
                'execution_time': results.execution_time,
                'num_spam_nodes': len(_extract_all_spam_nodes(results.spam_metadata)),
                'num_targets': len(_extract_target_nodes(results.spam_metadata))
            }
            
            # Add detection metrics if available
            if results.detection_evaluations:
                for method, eval_data in results.detection_evaluations.items():
                    if isinstance(eval_data, dict) and 'metrics' in eval_data:
                        metrics = eval_data['metrics']
                        summary[f'{method}_precision'] = metrics.get('precision', 0.0)
                        summary[f'{method}_recall'] = metrics.get('recall', 0.0)
                        summary[f'{method}_f1'] = metrics.get('f1_score', 0.0)
            
            # Add effectiveness metrics
            if results.effectiveness_analysis:
                if 'amplification' in results.effectiveness_analysis:
                    amp_data = results.effectiveness_analysis['amplification']
                    if amp_data:
                        summary['avg_amplification'] = np.mean([d.get('amplification', 0) for d in amp_data])
                
                if 'roi' in results.effectiveness_analysis:
                    roi_data = results.effectiveness_analysis['roi']
                    if roi_data:
                        summary['avg_roi'] = np.mean([d.get('roi', 0) for d in roi_data])
            
            return summary
        
        except Exception as e:
            logger.error(f"Error in experiment {combo}: {e}")
            return None
    
    # Run experiments
    if HAS_JOBLIB and n_jobs != 1:
        logger.info(f"Running {len(combinations)} experiments in parallel (n_jobs={n_jobs})...")
        summaries = Parallel(n_jobs=n_jobs)(
            delayed(run_single_experiment)(combo) 
            for combo in tqdm(combinations, desc="Parameter sweep")
        )
    else:
        logger.info(f"Running {len(combinations)} experiments sequentially...")
        summaries = [
            run_single_experiment(combo) 
            for combo in tqdm(combinations, desc="Parameter sweep")
        ]
    
    # Filter out None results
    summaries = [s for s in summaries if s is not None]
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summaries)
    
    # Save summary
    summary_path = os.path.join(output_dir, 'parameter_sweep_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Summary saved to {summary_path}")
    
    return summary_df


def compare_spam_strategies(
    G_original: nx.Graph,
    strategies_dict: Dict[str, Dict],
    output_dir: str
) -> Dict:
    """
    Compare different spam strategies.
    
    Args:
        G_original: Original graph
        strategies_dict: Dictionary mapping strategy names to configs
            Example: {
                'simple_large': {'spam_type': 'simple', 'm': 5000},
                'multiple_small': {'spam_type': 'multiple', 'm': 1000}
            }
        output_dir: Directory to save results
    
    Returns:
        Dictionary with comparison results
    """
    logger.info("=" * 60)
    logger.info(f"Comparing {len(strategies_dict)} spam strategies")
    logger.info("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    results_dict = {}
    
    for strategy_name, config in tqdm(strategies_dict.items(), desc="Strategies"):
        logger.info(f"Running strategy: {strategy_name}")
        
        try:
            results = run_complete_spam_experiment(
                G_original, config, save_intermediate=True, 
                output_dir=os.path.join(output_dir, strategy_name)
            )
            results_dict[strategy_name] = results
        
        except Exception as e:
            logger.error(f"Error in strategy {strategy_name}: {e}", exc_info=True)
            continue
    
    # Create comparison report
    comparison_data = []
    
    for strategy_name, results in results_dict.items():
        row = {'strategy': strategy_name}
        
        # Extract key metrics
        if results.detection_evaluations:
            best_f1 = 0.0
            best_method = None
            for method, eval_data in results.detection_evaluations.items():
                if isinstance(eval_data, dict) and 'metrics' in eval_data:
                    f1 = eval_data['metrics'].get('f1_score', 0.0)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_method = method
            row['best_detection_f1'] = best_f1
            row['best_detection_method'] = best_method
        
        if results.effectiveness_analysis:
            if 'amplification' in results.effectiveness_analysis:
                amp_data = results.effectiveness_analysis['amplification']
                if amp_data:
                    row['avg_amplification'] = np.mean([d.get('amplification', 0) for d in amp_data])
            
            if 'roi' in results.effectiveness_analysis:
                roi_data = results.effectiveness_analysis['roi']
                if roi_data:
                    row['avg_roi'] = np.mean([d.get('roi', 0) for d in roi_data])
        
        row['execution_time'] = results.execution_time
        row['num_spam_nodes'] = len(_extract_all_spam_nodes(results.spam_metadata))
        
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save comparison
    comparison_path = os.path.join(output_dir, 'strategy_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)
    logger.info(f"Comparison saved to {comparison_path}")
    
    # Identify best strategy for spammers (highest ROI, lowest detectability)
    if 'avg_roi' in comparison_df.columns and 'best_detection_f1' in comparison_df.columns:
        comparison_df['spammer_score'] = (
            comparison_df['avg_roi'] / (1 + comparison_df['best_detection_f1'])
        )
        best_spammer_strategy = comparison_df.loc[comparison_df['spammer_score'].idxmax(), 'strategy']
        logger.info(f"Best strategy for spammers: {best_spammer_strategy}")
    
    # Identify best detection approach
    all_detection_results = {}
    for strategy_name, results in results_dict.items():
        if results.detection_evaluations:
            for method, eval_data in results.detection_evaluations.items():
                if isinstance(eval_data, dict) and 'metrics' in eval_data:
                    if method not in all_detection_results:
                        all_detection_results[method] = []
                    all_detection_results[method].append(eval_data['metrics'].get('f1_score', 0.0))
    
    if all_detection_results:
        avg_f1_by_method = {method: np.mean(scores) for method, scores in all_detection_results.items()}
        best_detection_method = max(avg_f1_by_method.items(), key=lambda x: x[1])[0]
        logger.info(f"Best detection method: {best_detection_method} (avg F1: {avg_f1_by_method[best_detection_method]:.4f})")
    
    return {
        'comparison_df': comparison_df,
        'results_dict': results_dict,
        'best_spammer_strategy': best_spammer_strategy if 'best_spammer_strategy' in locals() else None,
        'best_detection_method': best_detection_method if 'best_detection_method' in locals() else None
    }


def generate_final_report(
    experiment_results_list: List[ExperimentResults],
    output_dir: str
) -> str:
    """
    Generate comprehensive final report.
    
    Args:
        experiment_results_list: List of experiment results
        output_dir: Directory to save report
    
    Returns:
        Path to generated HTML report
    """
    logger.info("Generating final comprehensive report...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Aggregate results
    all_summaries = []
    for results in experiment_results_list:
        summary = {
            'experiment_id': results.experiment_id,
            'spam_type': results.config.get('spam_type', 'unknown'),
            'm': results.config.get('m', 0),
            'execution_time': results.execution_time
        }
        
        if results.detection_evaluations:
            for method, eval_data in results.detection_evaluations.items():
                if isinstance(eval_data, dict) and 'metrics' in eval_data:
                    summary[f'{method}_f1'] = eval_data['metrics'].get('f1_score', 0.0)
        
        if results.effectiveness_analysis:
            if 'amplification' in results.effectiveness_analysis:
                amp_data = results.effectiveness_analysis['amplification']
                if amp_data:
                    summary['avg_amplification'] = np.mean([d.get('amplification', 0) for d in amp_data])
        
        all_summaries.append(summary)
    
    summary_df = pd.DataFrame(all_summaries)
    
    # Generate HTML report
    html_path = os.path.join(output_dir, 'final_report.html')
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Spam Analysis Final Report</title>
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
        </style>
    </head>
    <body>
        <h1>Spam Analysis Final Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Total Experiments:</strong> {len(experiment_results_list)}</p>
        
        <h2>Executive Summary</h2>
        <p>This report summarizes {len(experiment_results_list)} spam analysis experiments.</p>
        
        <h2>Experiment Summary</h2>
        {summary_df.to_html(index=False, classes='summary-table')}
        
        <h2>Key Findings</h2>
        <ul>
            <li>Best spam strategy: [Analysis needed]</li>
            <li>Best detection method: [Analysis needed]</li>
            <li>Average amplification: {summary_df['avg_amplification'].mean():.4f if 'avg_amplification' in summary_df.columns else 'N/A'}</li>
        </ul>
        
        <h2>Recommendations</h2>
        <h3>For Search Engines</h3>
        <ul>
            <li>Implement TrustRank with carefully selected trusted pages</li>
            <li>Monitor structural patterns (star, honeypot, reciprocal links)</li>
            <li>Use ensemble detection combining multiple methods</li>
            <li>Regularly update detection algorithms as spam evolves</li>
        </ul>
        
        <h3>For Further Research</h3>
        <ul>
            <li>Investigate adaptive spam strategies</li>
            <li>Study temporal patterns in spam injection</li>
            <li>Develop real-time detection systems</li>
            <li>Analyze economic incentives for spam</li>
        </ul>
    </body>
    </html>
    """
    
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Final report saved to {html_path}")
    
    return html_path


def create_demo_notebook(output_path: str):
    """
    Generate Jupyter notebook demonstrating spam analysis.
    
    Args:
        output_path: Path to save notebook (.ipynb file)
    """
    logger.info("Creating demo notebook...")
    
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Spam Farm Analysis Demo\n",
                    "\n",
                    "This notebook demonstrates the complete spam analysis pipeline."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "import sys\n",
                    "import os\n",
                    "sys.path.insert(0, 'src')\n",
                    "\n",
                    "import networkx as nx\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "from spam_analysis_pipeline import run_complete_spam_experiment"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Load or Create Sample Graph"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Create sample graph\n",
                    "G = nx.erdos_renyi_graph(1000, 0.01, seed=42)\n",
                    "print(f\"Graph: {len(G)} nodes, {G.number_of_edges()} edges\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Configure Experiment"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "experiment_config = {\n",
                    "    'spam_type': 'simple',\n",
                    "    'm': 500,  # supporting pages\n",
                    "    'external_links': 10,\n",
                    "    'beta': 0.85,\n",
                    "    'detection_methods': ['trustrank', 'spam_mass', 'structural'],\n",
                    "    'trusted_nodes': None  # Will auto-select\n",
                    "}"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Run Complete Experiment"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "results = run_complete_spam_experiment(\n",
                    "    G, \n",
                    "    experiment_config,\n",
                    "    save_intermediate=True,\n",
                    "    output_dir='results/demo'\n",
                    ")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Analyze Results"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Detection evaluation\n",
                    "if results.detection_evaluations:\n",
                    "    print(\"Detection Results:\")\n",
                    "    for method, eval_data in results.detection_evaluations.items():\n",
                    "        if isinstance(eval_data, dict) and 'metrics' in eval_data:\n",
                    "            metrics = eval_data['metrics']\n",
                    "            print(f\"\\n{method}:\")\n",
                    "            print(f\"  Precision: {metrics.get('precision', 0):.4f}\")\n",
                    "            print(f\"  Recall: {metrics.get('recall', 0):.4f}\")\n",
                    "            print(f\"  F1-Score: {metrics.get('f1_score', 0):.4f}\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Effectiveness analysis\n",
                    "if results.effectiveness_analysis:\n",
                    "    if 'amplification' in results.effectiveness_analysis:\n",
                    "        amp_df = pd.DataFrame(results.effectiveness_analysis['amplification'])\n",
                    "        print(\"\\nPageRank Amplification:\")\n",
                    "        print(amp_df[['target', 'amplification', 'predicted_amplification']])\n",
                    "    \n",
                    "    if 'roi' in results.effectiveness_analysis:\n",
                    "        roi_df = pd.DataFrame(results.effectiveness_analysis['roi'])\n",
                    "        print(\"\\nROI Analysis:\")\n",
                    "        print(roi_df[['target', 'cost', 'benefit', 'roi']])"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Visualizations"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "from spam_effectiveness_analysis import visualize_spam_impact\n",
                    "\n",
                    "plot_paths = visualize_spam_impact(\n",
                    "    results.G_original if hasattr(results, 'G_original') else G,\n",
                    "    results.G_with_spam,\n",
                    "    results.spam_metadata,\n",
                    "    'results/demo/plots'\n",
                    ")\n",
                    "\n",
                    "print(\"Plots saved to:\")\n",
                    "for name, path in plot_paths.items():\n",
                    "    print(f\"  {name}: {path}\")"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    import json
    with open(output_path, 'w') as f:
        json.dump(notebook_content, f, indent=2)
    
    logger.info(f"Demo notebook saved to {output_path}")


def main():
    """
    Main function demonstrating spam analysis pipeline.
    """
    print("=" * 60)
    print("Spam Analysis Pipeline - Demo")
    print("=" * 60)
    
    try:
        import networkx as nx
        
        # Create sample graph
        print("\n1. Creating sample graph...")
        G = nx.erdos_renyi_graph(500, 0.02, seed=42)
        print(f"   Nodes: {len(G)}, Edges: {G.number_of_edges()}")
        
        # Run single experiment
        print("\n2. Running complete spam experiment...")
        config = {
            'spam_type': 'simple',
            'm': 100,
            'external_links': 5,
            'beta': 0.85,
            'detection_methods': ['trustrank', 'spam_mass'],
            'trusted_nodes': None
        }
        
        results = run_complete_spam_experiment(
            G, config, save_intermediate=False, output_dir=None
        )
        
        print(f"   Experiment ID: {results.experiment_id}")
        print(f"   Execution time: {results.execution_time:.2f}s")
        print(f"   Spam nodes: {len(_extract_all_spam_nodes(results.spam_metadata))}")
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)
    
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"\n✗ Error: {e}")
        raise


if __name__ == "__main__":
    main()

