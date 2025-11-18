"""
Link prediction module for network graphs.

This module provides functions for predicting missing links in networks
using similarity-based methods including Common Neighbors, Adamic-Adar,
and Jaccard Coefficient.
"""

import logging
import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def common_neighbors_score(G: nx.Graph, edge_list: List[Tuple[int, int]]) -> List[float]:
    """
    Calculate Common Neighbors score for a list of edges.
    
    Common Neighbors score is the number of neighbors shared by two nodes.
    Higher scores indicate higher likelihood of a link.
    
    Args:
        G: NetworkX graph
        edge_list: List of (u, v) tuples representing edges to score
    
    Returns:
        List of scores (one per edge)
    
    Raises:
        TypeError: If G is not a NetworkX graph
        ValueError: If edge_list is empty
    """
    if not isinstance(G, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise TypeError(f"Expected NetworkX graph, got {type(G)}")
    
    if not edge_list:
        raise ValueError("edge_list is empty")
    
    logger.info(f"Calculating Common Neighbors scores for {len(edge_list)} edges...")
    
    scores = []
    batch_size = 10000  # Process in batches for memory efficiency
    
    for i in tqdm(range(0, len(edge_list), batch_size), desc="Processing batches"):
        batch = edge_list[i:i+batch_size]
        batch_scores = []
        
        for u, v in batch:
            if u in G and v in G:
                # Get neighbors of u and v
                neighbors_u = set(G.neighbors(u))
                neighbors_v = set(G.neighbors(v))
                # Common neighbors count
                common = len(neighbors_u & neighbors_v)
                batch_scores.append(float(common))
            else:
                # If node not in graph, score is 0
                batch_scores.append(0.0)
        
        scores.extend(batch_scores)
    
    logger.info(f"Common Neighbors scores calculated: {len(scores)} scores")
    return scores


def adamic_adar_score(G: nx.Graph, edge_list: List[Tuple[int, int]]) -> List[float]:
    """
    Calculate Adamic-Adar score for a list of edges.
    
    Adamic-Adar score weights common neighbors by the inverse logarithm
    of their degree. Nodes with fewer neighbors contribute more.
    
    Args:
        G: NetworkX graph
        edge_list: List of (u, v) tuples representing edges to score
    
    Returns:
        List of scores (one per edge)
    
    Raises:
        TypeError: If G is not a NetworkX graph
        ValueError: If edge_list is empty
    """
    if not isinstance(G, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise TypeError(f"Expected NetworkX graph, got {type(G)}")
    
    if not edge_list:
        raise ValueError("edge_list is empty")
    
    logger.info(f"Calculating Adamic-Adar scores for {len(edge_list)} edges...")
    
    # Use NetworkX function if available
    try:
        # Convert to generator for NetworkX
        scores = []
        batch_size = 10000
        
        for i in tqdm(range(0, len(edge_list), batch_size), desc="Processing batches"):
            batch = edge_list[i:i+batch_size]
            
            # Use NetworkX adamic_adar_index
            aa_scores = nx.adamic_adar_index(G, batch)
            batch_scores = [score for _, _, score in aa_scores]
            scores.extend(batch_scores)
        
        logger.info(f"Adamic-Adar scores calculated: {len(scores)} scores")
        return scores
    
    except Exception as e:
        logger.error(f"Failed to calculate Adamic-Adar scores: {e}")
        raise


def jaccard_coefficient_score(G: nx.Graph, edge_list: List[Tuple[int, int]]) -> List[float]:
    """
    Calculate Jaccard Coefficient score for a list of edges.
    
    Jaccard Coefficient is the ratio of common neighbors to total unique neighbors.
    Range: [0, 1] where 1 indicates all neighbors are shared.
    
    Args:
        G: NetworkX graph
        edge_list: List of (u, v) tuples representing edges to score
    
    Returns:
        List of scores (one per edge)
    
    Raises:
        TypeError: If G is not a NetworkX graph
        ValueError: If edge_list is empty
    """
    if not isinstance(G, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise TypeError(f"Expected NetworkX graph, got {type(G)}")
    
    if not edge_list:
        raise ValueError("edge_list is empty")
    
    logger.info(f"Calculating Jaccard Coefficient scores for {len(edge_list)} edges...")
    
    # Use NetworkX function if available
    try:
        scores = []
        batch_size = 10000
        
        for i in tqdm(range(0, len(edge_list), batch_size), desc="Processing batches"):
            batch = edge_list[i:i+batch_size]
            
            # Use NetworkX jaccard_coefficient
            jaccard_scores = nx.jaccard_coefficient(G, batch)
            batch_scores = [score for _, _, score in jaccard_scores]
            scores.extend(batch_scores)
        
        logger.info(f"Jaccard Coefficient scores calculated: {len(scores)} scores")
        return scores
    
    except Exception as e:
        logger.error(f"Failed to calculate Jaccard Coefficient scores: {e}")
        raise


def calculate_all_similarity_scores(
    G: nx.Graph,
    edge_list: List[Tuple[int, int]]
) -> pd.DataFrame:
    """
    Calculate all three similarity measures for a list of edges.
    
    Computes Common Neighbors, Adamic-Adar, and Jaccard Coefficient
    scores for all edges in the list.
    
    Args:
        G: NetworkX graph
        edge_list: List of (u, v) tuples representing edges to score
    
    Returns:
        DataFrame with columns: [node1, node2, common_neighbors, adamic_adar, jaccard]
    
    Raises:
        TypeError: If G is not a NetworkX graph
        ValueError: If edge_list is empty
    """
    if not isinstance(G, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise TypeError(f"Expected NetworkX graph, got {type(G)}")
    
    if not edge_list:
        raise ValueError("edge_list is empty")
    
    logger.info(f"Calculating all similarity scores for {len(edge_list)} edges...")
    
    # Calculate all scores
    cn_scores = common_neighbors_score(G, edge_list)
    aa_scores = adamic_adar_score(G, edge_list)
    jc_scores = jaccard_coefficient_score(G, edge_list)
    
    # Create DataFrame
    df = pd.DataFrame({
        'node1': [u for u, v in edge_list],
        'node2': [v for u, v in edge_list],
        'common_neighbors': cn_scores,
        'adamic_adar': aa_scores,
        'jaccard': jc_scores
    })
    
    logger.info("All similarity scores calculated successfully")
    return df


def predict_links_similarity(
    G: nx.Graph,
    method: str = 'adamic_adar',
    threshold: Optional[float] = None,
    top_k: Optional[int] = None,
    max_candidates: Optional[int] = 100000
) -> pd.DataFrame:
    """
    Predict missing links using similarity-based methods.
    
    Generates candidate edges (non-edges) and calculates similarity scores.
    Returns top predictions based on threshold or top-k.
    
    Args:
        G: NetworkX graph
        method: Similarity method - 'common_neighbors', 'adamic_adar', or 'jaccard' (default: 'adamic_adar')
        threshold: Score threshold for predictions (default: None, uses top-k)
        top_k: Number of top predictions to return (default: None, uses threshold)
        max_candidates: Maximum number of candidate edges to consider (default: 100000)
    
    Returns:
        DataFrame with columns: [node1, node2, score] sorted by score descending
    
    Raises:
        TypeError: If G is not a NetworkX graph
        ValueError: If method is invalid or parameters are invalid
    """
    if not isinstance(G, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise TypeError(f"Expected NetworkX graph, got {type(G)}")
    
    valid_methods = ['common_neighbors', 'adamic_adar', 'jaccard']
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got {method}")
    
    if threshold is None and top_k is None:
        raise ValueError("Either threshold or top_k must be specified")
    
    logger.info(f"Predicting links using {method} method...")
    
    # Generate candidate edges (non-edges)
    logger.info("Generating candidate edges (non-edges)...")
    nodes = list(G.nodes())
    num_nodes = len(nodes)
    existing_edges = set(G.edges())
    
    # Sample candidate edges if graph is too large
    if num_nodes * (num_nodes - 1) / 2 > max_candidates:
        logger.info(f"Graph is large, sampling {max_candidates} candidate edges...")
        candidates = []
        attempts = 0
        max_attempts = max_candidates * 10
        
        while len(candidates) < max_candidates and attempts < max_attempts:
            attempts += 1
            u, v = np.random.choice(nodes, size=2, replace=False)
            if u > v:
                u, v = v, u
            edge = (u, v)
            if edge not in existing_edges:
                candidates.append(edge)
    else:
        # Generate all non-edges
        candidates = []
        for i, u in enumerate(tqdm(nodes, desc="Generating candidates")):
            for v in nodes[i+1:]:
                if (u, v) not in existing_edges and (v, u) not in existing_edges:
                    candidates.append((u, v))
    
    logger.info(f"Generated {len(candidates)} candidate edges")
    
    # Calculate scores
    if method == 'common_neighbors':
        scores = common_neighbors_score(G, candidates)
    elif method == 'adamic_adar':
        scores = adamic_adar_score(G, candidates)
    elif method == 'jaccard':
        scores = jaccard_coefficient_score(G, candidates)
    
    # Create DataFrame
    df = pd.DataFrame({
        'node1': [u for u, v in candidates],
        'node2': [v for u, v in candidates],
        'score': scores
    })
    
    # Filter and sort
    if threshold is not None:
        df = df[df['score'] >= threshold]
        df = df.sort_values('score', ascending=False)
        logger.info(f"Filtered to {len(df)} predictions above threshold {threshold}")
    elif top_k is not None:
        df = df.sort_values('score', ascending=False).head(top_k)
        logger.info(f"Selected top {len(df)} predictions")
    
    return df


def evaluate_link_prediction(
    G_train: nx.Graph,
    pos_test: List[Tuple[int, int]],
    neg_test: List[Tuple[int, int]],
    method: str = 'adamic_adar'
) -> Dict[str, float]:
    """
    Evaluate link prediction performance using similarity-based methods.
    
    Calculates scores for positive and negative test edges, determines
    optimal threshold, and computes various evaluation metrics.
    
    Args:
        G_train: Training graph (used for prediction)
        pos_test: List of positive test edges (u, v) tuples
        neg_test: List of negative test edges (u, v) tuples
        method: Similarity method - 'common_neighbors', 'adamic_adar', or 'jaccard' (default: 'adamic_adar')
    
    Returns:
        Dictionary containing:
            - 'precision': Precision score
            - 'recall': Recall score
            - 'f1': F1 score
            - 'auc_roc': AUC-ROC score
            - 'auc_pr': AUC-PR (Average Precision) score
            - 'optimal_threshold': Optimal threshold value
            - 'num_pos_predictions': Number of positive predictions at optimal threshold
    
    Raises:
        TypeError: If inputs are not of correct types
        ValueError: If inputs are invalid
    """
    if not isinstance(G_train, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise TypeError(f"Expected NetworkX graph for G_train, got {type(G_train)}")
    
    if not isinstance(pos_test, list) or not isinstance(neg_test, list):
        raise TypeError("pos_test and neg_test must be lists")
    
    if not pos_test or not neg_test:
        raise ValueError("pos_test and neg_test must not be empty")
    
    valid_methods = ['common_neighbors', 'adamic_adar', 'jaccard']
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got {method}")
    
    logger.info(f"Evaluating link prediction using {method} method...")
    logger.info(f"Positive test edges: {len(pos_test)}, Negative test edges: {len(neg_test)}")
    
    # Calculate scores for positive test edges
    logger.info("Calculating scores for positive test edges...")
    if method == 'common_neighbors':
        pos_scores = common_neighbors_score(G_train, pos_test)
    elif method == 'adamic_adar':
        pos_scores = adamic_adar_score(G_train, pos_test)
    elif method == 'jaccard':
        pos_scores = jaccard_coefficient_score(G_train, pos_test)
    
    # Calculate scores for negative test edges
    logger.info("Calculating scores for negative test edges...")
    if method == 'common_neighbors':
        neg_scores = common_neighbors_score(G_train, neg_test)
    elif method == 'adamic_adar':
        neg_scores = adamic_adar_score(G_train, neg_test)
    elif method == 'jaccard':
        neg_scores = jaccard_coefficient_score(G_train, neg_test)
    
    # Combine scores and labels
    all_scores = np.array(pos_scores + neg_scores)
    all_labels = np.array([1] * len(pos_scores) + [0] * len(neg_scores))
    
    # Calculate AUC-ROC
    try:
        auc_roc = roc_auc_score(all_labels, all_scores)
    except Exception as e:
        logger.warning(f"Could not calculate AUC-ROC: {e}")
        auc_roc = np.nan
    
    # Calculate AUC-PR (Average Precision)
    try:
        auc_pr = average_precision_score(all_labels, all_scores)
    except Exception as e:
        logger.warning(f"Could not calculate AUC-PR: {e}")
        auc_pr = np.nan
    
    # Find optimal threshold (maximizing F1)
    logger.info("Finding optimal threshold...")
    thresholds = np.linspace(all_scores.min(), all_scores.max(), 100)
    best_f1 = -1
    optimal_threshold = 0.0
    best_precision = 0.0
    best_recall = 0.0
    
    for threshold in thresholds:
        predictions = (all_scores >= threshold).astype(int)
        try:
            precision = precision_score(all_labels, predictions, zero_division=0)
            recall = recall_score(all_labels, predictions, zero_division=0)
            f1 = f1_score(all_labels, predictions, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                optimal_threshold = threshold
                best_precision = precision
                best_recall = recall
        except:
            continue
    
    num_pos_predictions = np.sum(all_scores >= optimal_threshold)
    
    metrics = {
        'precision': best_precision,
        'recall': best_recall,
        'f1': best_f1,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'optimal_threshold': optimal_threshold,
        'num_pos_predictions': int(num_pos_predictions)
    }
    
    logger.info(f"Evaluation complete: F1={best_f1:.4f}, AUC-ROC={auc_roc:.4f}, AUC-PR={auc_pr:.4f}")
    
    return metrics


def main():
    """
    Main function demonstrating usage of the link prediction module.
    """
    print("=" * 60)
    print("Link Prediction Module - Demo")
    print("=" * 60)
    
    try:
        import networkx as nx
        
        # Create a sample graph
        print("\n1. Creating sample graph...")
        G = nx.erdos_renyi_graph(n=1000, p=0.01, seed=42)
        print(f"   Created graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Create train/test split
        print("\n2. Creating train/test split...")
        from preprocessing import create_train_test_split
        G_train, pos_test, neg_test = create_train_test_split(G, test_ratio=0.2, seed=42)
        print(f"   Train: {G_train.number_of_edges()} edges")
        print(f"   Test: {len(pos_test)} positive + {len(neg_test)} negative edges")
        
        # Calculate all similarity scores for test edges
        print("\n3. Calculating similarity scores...")
        all_test_edges = pos_test + neg_test
        scores_df = calculate_all_similarity_scores(G_train, all_test_edges[:100])  # Sample for demo
        print(f"   Calculated scores for {len(scores_df)} edges")
        print(scores_df.head(10).to_string(index=False))
        
        # Evaluate link prediction
        print("\n4. Evaluating link prediction...")
        metrics = evaluate_link_prediction(G_train, pos_test[:100], neg_test[:100], method='adamic_adar')
        print("   Evaluation metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"     {key}: {value:.4f}")
            else:
                print(f"     {key}: {value}")
        
        # Predict links
        print("\n5. Predicting top links...")
        predictions = predict_links_similarity(G_train, method='adamic_adar', top_k=10)
        print(f"   Top 10 predicted links:")
        print(predictions.to_string(index=False))
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)
    
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"\n✗ Error: {e}")
        raise


if __name__ == "__main__":
    main()

