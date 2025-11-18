"""
Community evaluation module for network graphs.

This module provides functions for evaluating community detection results
against ground truth using various metrics including NMI, ARI, and modularity.
"""

import os
import logging
import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Set, Tuple, Optional
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score
)

# Import community detection module functions
from community_detection import (
    calculate_modularity,
    communities_to_dict,
    get_community_sizes
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_ground_truth_communities(filepath: str) -> Dict[int, int]:
    """
    Load ground-truth communities from file and convert to node-to-community mapping.
    
    Expected format: Each line contains space-separated node IDs representing a community.
    
    Args:
        filepath: Path to the ground-truth communities file
    
    Returns:
        Dictionary mapping node ID to community ID (0-indexed)
    
    Raises:
        FileNotFoundError: If file doesn't exist
        IOError: If file cannot be read
        ValueError: If file format is invalid
    """
    import gzip
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Ground truth file not found: {filepath}")
    
    logger.info(f"Loading ground-truth communities from {filepath}")
    
    # Determine if file is gzipped
    is_gzipped = filepath.endswith('.gz')
    
    try:
        # Open file (gzipped or regular)
        if is_gzipped:
            open_func = gzip.open
            mode = 'rt'
        else:
            open_func = open
            mode = 'r'
        
        node_to_community = {}
        community_id = 0
        
        with open_func(filepath, mode) as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Parse community (space-separated node IDs)
                try:
                    nodes = [int(node_id) for node_id in line.split()]
                    if nodes:  # Only process non-empty communities
                        for node in nodes:
                            if node in node_to_community:
                                logger.warning(f"Node {node} appears in multiple ground-truth communities")
                            node_to_community[node] = community_id
                        community_id += 1
                except ValueError as e:
                    logger.warning(f"Skipping invalid line {line_num}: {e}")
                    continue
        
        logger.info(f"Loaded {community_id} ground-truth communities with {len(node_to_community)} nodes")
        return node_to_community
    
    except IOError as e:
        logger.error(f"Failed to read file {filepath}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while loading ground truth: {e}")
        raise ValueError(f"Failed to load ground truth from {filepath}: {e}")


def calculate_nmi(
    detected_communities: List[Set],
    ground_truth: Dict[int, int]
) -> float:
    """
    Calculate Normalized Mutual Information (NMI) score.
    
    NMI measures the agreement between detected and ground-truth communities.
    Range: [0, 1] where 1 indicates perfect agreement.
    
    Args:
        detected_communities: List of sets, each set contains node IDs in a detected community
        ground_truth: Dictionary mapping node ID to ground-truth community ID
    
    Returns:
        NMI score (float between 0 and 1)
    
    Raises:
        ValueError: If inputs are invalid or empty
    """
    if not detected_communities:
        raise ValueError("Detected communities list is empty")
    
    if not ground_truth:
        raise ValueError("Ground truth is empty")
    
    logger.info("Calculating NMI score...")
    
    # Convert detected communities to node-to-community mapping
    detected_mapping = communities_to_dict(detected_communities)
    
    # Find common nodes
    common_nodes = set(detected_mapping.keys()) & set(ground_truth.keys())
    
    if not common_nodes:
        logger.warning("No common nodes between detected and ground truth")
        return 0.0
    
    if len(common_nodes) < 2:
        logger.warning("Less than 2 common nodes, NMI may be unreliable")
        return 0.0
    
    # Create label arrays for common nodes only
    detected_labels = [detected_mapping[node] for node in sorted(common_nodes)]
    ground_truth_labels = [ground_truth[node] for node in sorted(common_nodes)]
    
    # Calculate NMI
    try:
        nmi = normalized_mutual_info_score(ground_truth_labels, detected_labels)
        logger.info(f"NMI score: {nmi:.6f} (based on {len(common_nodes)} common nodes)")
        return nmi
    except Exception as e:
        logger.error(f"Failed to calculate NMI: {e}")
        raise


def calculate_ari(
    detected_communities: List[Set],
    ground_truth: Dict[int, int]
) -> float:
    """
    Calculate Adjusted Rand Index (ARI) score.
    
    ARI measures the similarity between two clusterings, adjusted for chance.
    Range: [-1, 1] where 1 indicates perfect agreement, 0 indicates random agreement.
    
    Args:
        detected_communities: List of sets, each set contains node IDs in a detected community
        ground_truth: Dictionary mapping node ID to ground-truth community ID
    
    Returns:
        ARI score (float between -1 and 1)
    
    Raises:
        ValueError: If inputs are invalid or empty
    """
    if not detected_communities:
        raise ValueError("Detected communities list is empty")
    
    if not ground_truth:
        raise ValueError("Ground truth is empty")
    
    logger.info("Calculating ARI score...")
    
    # Convert detected communities to node-to-community mapping
    detected_mapping = communities_to_dict(detected_communities)
    
    # Find common nodes
    common_nodes = set(detected_mapping.keys()) & set(ground_truth.keys())
    
    if not common_nodes:
        logger.warning("No common nodes between detected and ground truth")
        return 0.0
    
    if len(common_nodes) < 2:
        logger.warning("Less than 2 common nodes, ARI may be unreliable")
        return 0.0
    
    # Create label arrays for common nodes only
    detected_labels = [detected_mapping[node] for node in sorted(common_nodes)]
    ground_truth_labels = [ground_truth[node] for node in sorted(common_nodes)]
    
    # Calculate ARI
    try:
        ari = adjusted_rand_score(ground_truth_labels, detected_labels)
        logger.info(f"ARI score: {ari:.6f} (based on {len(common_nodes)} common nodes)")
        return ari
    except Exception as e:
        logger.error(f"Failed to calculate ARI: {e}")
        raise


def evaluate_communities(
    detected: List[Set],
    ground_truth: Dict[int, int],
    G: nx.Graph
) -> Dict[str, float]:
    """
    Evaluate detected communities against ground truth.
    
    Calculates multiple evaluation metrics:
    - NMI (Normalized Mutual Information)
    - ARI (Adjusted Rand Index)
    - Modularity
    - Coverage (percentage of nodes covered)
    - Performance (fraction of correctly classified node pairs)
    
    Args:
        detected: List of sets, each set contains node IDs in a detected community
        ground_truth: Dictionary mapping node ID to ground-truth community ID
        G: NetworkX graph
    
    Returns:
        Dictionary containing all evaluation metrics
    
    Raises:
        TypeError: If inputs are not of correct types
        ValueError: If inputs are invalid
    """
    if not isinstance(detected, list):
        raise TypeError(f"Expected list for detected, got {type(detected)}")
    
    if not isinstance(ground_truth, dict):
        raise TypeError(f"Expected dict for ground_truth, got {type(ground_truth)}")
    
    if not isinstance(G, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise TypeError(f"Expected NetworkX graph, got {type(G)}")
    
    logger.info("Evaluating communities...")
    
    metrics = {}
    
    # Calculate NMI
    try:
        metrics['nmi'] = calculate_nmi(detected, ground_truth)
    except Exception as e:
        logger.warning(f"Could not calculate NMI: {e}")
        metrics['nmi'] = np.nan
    
    # Calculate ARI
    try:
        metrics['ari'] = calculate_ari(detected, ground_truth)
    except Exception as e:
        logger.warning(f"Could not calculate ARI: {e}")
        metrics['ari'] = np.nan
    
    # Calculate Modularity
    try:
        metrics['modularity'] = calculate_modularity(G, detected)
    except Exception as e:
        logger.warning(f"Could not calculate modularity: {e}")
        metrics['modularity'] = np.nan
    
    # Calculate Coverage
    detected_mapping = communities_to_dict(detected)
    detected_nodes = set(detected_mapping.keys())
    ground_truth_nodes = set(ground_truth.keys())
    all_nodes = detected_nodes | ground_truth_nodes
    
    if all_nodes:
        coverage = len(detected_nodes & ground_truth_nodes) / len(all_nodes)
        metrics['coverage'] = coverage
    else:
        metrics['coverage'] = 0.0
    
    # Calculate Performance (fraction of correctly classified node pairs)
    try:
        detected_mapping = communities_to_dict(detected)
        common_nodes = set(detected_mapping.keys()) & set(ground_truth.keys())
        
        if len(common_nodes) < 2:
            metrics['performance'] = 0.0
        else:
            correct_pairs = 0
            total_pairs = 0
            
            common_nodes_list = sorted(common_nodes)
            for i, node1 in enumerate(common_nodes_list):
                for node2 in common_nodes_list[i+1:]:
                    total_pairs += 1
                    # Check if both nodes are in same community in both partitions
                    same_detected = (detected_mapping[node1] == detected_mapping[node2])
                    same_ground_truth = (ground_truth[node1] == ground_truth[node2])
                    if same_detected == same_ground_truth:
                        correct_pairs += 1
            
            metrics['performance'] = correct_pairs / total_pairs if total_pairs > 0 else 0.0
    except Exception as e:
        logger.warning(f"Could not calculate performance: {e}")
        metrics['performance'] = np.nan
    
    # Additional statistics
    metrics['num_detected_communities'] = len(detected)
    metrics['num_ground_truth_communities'] = len(set(ground_truth.values()))
    metrics['num_detected_nodes'] = len(detected_mapping)
    metrics['num_ground_truth_nodes'] = len(ground_truth)
    metrics['num_common_nodes'] = len(common_nodes)
    
    logger.info(f"Evaluation complete: NMI={metrics['nmi']:.4f}, ARI={metrics['ari']:.4f}, "
               f"Modularity={metrics['modularity']:.4f}")
    
    return metrics


def compare_all_methods(
    detection_results: Dict,
    ground_truth: Dict[int, int],
    G: nx.Graph
) -> pd.DataFrame:
    """
    Evaluate all community detection methods and create comparison table.
    
    Args:
        detection_results: Dictionary from run_all_community_detection()
        ground_truth: Dictionary mapping node ID to ground-truth community ID
        G: NetworkX graph
    
    Returns:
        Pandas DataFrame with columns:
            - Method: Algorithm name
            - Num_Communities: Number of detected communities
            - Modularity: Modularity score
            - NMI: Normalized Mutual Information score
            - ARI: Adjusted Rand Index score
            - Coverage: Node coverage percentage
            - Performance: Classification performance
            - Runtime: Execution time in seconds
    
    Raises:
        TypeError: If inputs are not of correct types
        ValueError: If inputs are invalid
    """
    if not isinstance(detection_results, dict):
        raise TypeError(f"Expected dict for detection_results, got {type(detection_results)}")
    
    if not isinstance(ground_truth, dict):
        raise TypeError(f"Expected dict for ground_truth, got {type(ground_truth)}")
    
    if not isinstance(G, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise TypeError(f"Expected NetworkX graph, got {type(G)}")
    
    logger.info("Comparing all community detection methods...")
    
    comparison_data = []
    
    # Get timing information
    timing = detection_results.get('timing', {})
    
    # Evaluate each method
    method_names = ['louvain', 'label_propagation', 'greedy_modularity']
    
    for method_name in method_names:
        if method_name in detection_results and detection_results[method_name] is not None:
            detected_communities, _ = detection_results[method_name]
            
            # Evaluate
            metrics = evaluate_communities(detected_communities, ground_truth, G)
            
            # Add method name and timing
            metrics['method'] = method_name.replace('_', ' ').title()
            metrics['runtime'] = timing.get(method_name, np.nan)
            
            comparison_data.append(metrics)
        else:
            logger.warning(f"Method {method_name} not available in results")
    
    # Create DataFrame
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        
        # Reorder columns
        column_order = [
            'method', 'num_detected_communities', 'modularity', 'nmi', 'ari',
            'coverage', 'performance', 'runtime'
        ]
        # Only include columns that exist
        column_order = [col for col in column_order if col in df.columns]
        df = df[column_order]
        
        # Rename columns for better display
        df = df.rename(columns={
            'method': 'Method',
            'num_detected_communities': 'Num_Communities',
            'modularity': 'Modularity',
            'nmi': 'NMI',
            'ari': 'ARI',
            'coverage': 'Coverage',
            'performance': 'Performance',
            'runtime': 'Runtime (s)'
        })
        
        logger.info("Comparison table created successfully")
        return df
    else:
        logger.warning("No valid detection results to compare")
        return pd.DataFrame()


def analyze_community_overlap(
    detected_communities: List[Set],
    ground_truth: Dict[int, int]
) -> Dict:
    """
    Analyze overlap between detected and ground-truth communities.
    
    Finds best matching between detected communities and ground-truth communities,
    and calculates overlap statistics.
    
    Args:
        detected_communities: List of sets, each set contains node IDs in a detected community
        ground_truth: Dictionary mapping node ID to ground-truth community ID
    
    Returns:
        Dictionary containing:
            - best_matches: Dict mapping detected community index to best matching ground-truth community
            - overlap_matrix: Matrix of overlaps between detected and ground-truth communities
            - overlap_statistics: Statistics about overlaps
            - unmatched_detected: List of detected communities with no good match
            - unmatched_ground_truth: List of ground-truth communities with no good match
    
    Raises:
        ValueError: If inputs are invalid
    """
    if not detected_communities:
        raise ValueError("Detected communities list is empty")
    
    if not ground_truth:
        raise ValueError("Ground truth is empty")
    
    logger.info("Analyzing community overlap...")
    
    # Convert ground truth to list of sets
    ground_truth_communities = {}
    for node, comm_id in ground_truth.items():
        if comm_id not in ground_truth_communities:
            ground_truth_communities[comm_id] = set()
        ground_truth_communities[comm_id].add(node)
    ground_truth_list = list(ground_truth_communities.values())
    
    # Calculate overlap matrix
    overlap_matrix = np.zeros((len(detected_communities), len(ground_truth_list)))
    
    for i, detected_comm in enumerate(detected_communities):
        for j, gt_comm in enumerate(ground_truth_list):
            overlap = len(detected_comm & gt_comm)
            union = len(detected_comm | gt_comm)
            # Jaccard similarity
            jaccard = overlap / union if union > 0 else 0.0
            overlap_matrix[i, j] = jaccard
    
    # Find best matches (greedy matching)
    best_matches = {}
    used_gt = set()
    
    # Sort by overlap strength
    matches = []
    for i in range(len(detected_communities)):
        for j in range(len(ground_truth_list)):
            if overlap_matrix[i, j] > 0:
                matches.append((i, j, overlap_matrix[i, j]))
    
    matches.sort(key=lambda x: x[2], reverse=True)
    
    for i, j, score in matches:
        if i not in best_matches and j not in used_gt:
            best_matches[i] = j
            used_gt.add(j)
    
    # Find unmatched communities
    unmatched_detected = [i for i in range(len(detected_communities)) if i not in best_matches]
    unmatched_ground_truth = [j for j in range(len(ground_truth_list)) if j not in used_gt]
    
    # Calculate statistics
    if best_matches:
        match_scores = [overlap_matrix[i, j] for i, j in best_matches.items()]
        overlap_statistics = {
            'num_matches': len(best_matches),
            'avg_match_score': np.mean(match_scores),
            'max_match_score': np.max(match_scores),
            'min_match_score': np.min(match_scores),
            'num_unmatched_detected': len(unmatched_detected),
            'num_unmatched_ground_truth': len(unmatched_ground_truth)
        }
    else:
        overlap_statistics = {
            'num_matches': 0,
            'avg_match_score': 0.0,
            'max_match_score': 0.0,
            'min_match_score': 0.0,
            'num_unmatched_detected': len(unmatched_detected),
            'num_unmatched_ground_truth': len(unmatched_ground_truth)
        }
    
    results = {
        'best_matches': best_matches,
        'overlap_matrix': overlap_matrix,
        'overlap_statistics': overlap_statistics,
        'unmatched_detected': unmatched_detected,
        'unmatched_ground_truth': unmatched_ground_truth
    }
    
    logger.info(f"Overlap analysis complete: {overlap_statistics['num_matches']} matches found")
    
    return results


def main():
    """
    Main function demonstrating usage of the community evaluation module.
    """
    print("=" * 60)
    print("Community Evaluation Module - Demo")
    print("=" * 60)
    
    try:
        import networkx as nx
        from community_detection import run_all_community_detection
        
        # Create a sample graph with known community structure
        print("\n1. Creating sample graph with community structure...")
        G = nx.planted_partition_graph(4, 50, 0.5, 0.1, seed=42)
        print(f"   Created graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Create ground truth
        print("\n2. Creating ground truth...")
        ground_truth = {}
        for i in range(4):
            for j in range(50):
                node = i * 50 + j
                ground_truth[node] = i
        print(f"   Ground truth: {len(set(ground_truth.values()))} communities")
        
        # Run detection
        print("\n3. Running community detection...")
        detection_results = run_all_community_detection(G, seed=42)
        
        # Compare methods
        print("\n4. Comparing all methods...")
        comparison_df = compare_all_methods(detection_results, ground_truth, G)
        print("\nComparison Table:")
        print(comparison_df.to_string(index=False))
        
        # Analyze overlap
        if 'louvain' in detection_results and detection_results['louvain'] is not None:
            print("\n5. Analyzing community overlap...")
            detected_comm, _ = detection_results['louvain']
            overlap_results = analyze_community_overlap(detected_comm, ground_truth)
            print(f"   Matches: {overlap_results['overlap_statistics']['num_matches']}")
            print(f"   Avg match score: {overlap_results['overlap_statistics']['avg_match_score']:.4f}")
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)
    
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"\n✗ Error: {e}")
        raise


if __name__ == "__main__":
    import os
    main()

