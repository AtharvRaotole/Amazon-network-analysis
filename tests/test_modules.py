"""
Unit tests for network analysis modules.

This module contains comprehensive unit tests for all analysis modules
using pytest framework.
"""

import pytest
import sys
import os
from pathlib import Path
import pickle
import networkx as nx
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Import modules to test
from data_loader import load_graph, load_communities, save_graph, load_saved_graph
from preprocessing import (
    remove_self_loops,
    get_largest_component,
    create_train_test_split,
    basic_statistics
)
from centrality_analysis import (
    compute_pagerank,
    compute_degree_centrality,
    compute_betweenness_centrality,
    compute_hits,
    get_top_k_nodes
)
from community_detection import (
    detect_communities_louvain,
    detect_communities_label_propagation,
    detect_communities_greedy_modularity,
    calculate_modularity,
    get_community_sizes
)
from link_prediction import (
    common_neighbors_score,
    adamic_adar_score,
    jaccard_coefficient_score,
    evaluate_link_prediction
)
from ml_link_prediction import (
    extract_edge_features,
    prepare_training_data,
    train_random_forest,
    evaluate_ml_model
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_graph():
    """Create a sample undirected graph for testing."""
    G = nx.erdos_renyi_graph(n=100, p=0.1, seed=42)
    return G


@pytest.fixture
def sample_directed_graph():
    """Create a sample directed graph for testing."""
    G = nx.erdos_renyi_graph(n=50, p=0.15, seed=42, directed=True)
    return G


@pytest.fixture
def sample_graph_with_selfloops():
    """Create a graph with self-loops for testing."""
    G = nx.erdos_renyi_graph(n=50, p=0.1, seed=42)
    # Add some self-loops
    for node in list(G.nodes())[:5]:
        G.add_edge(node, node)
    return G


@pytest.fixture
def sample_communities():
    """Create sample ground-truth communities."""
    communities = {
        0: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2,
        6: 3, 7: 3, 8: 3, 9: 3
    }
    return communities


@pytest.fixture
def sample_edge_list():
    """Create sample edge list for testing."""
    return [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]


@pytest.fixture
def test_data_dir(tmp_path):
    """Create temporary directory for test data."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    return data_dir


# ============================================================================
# DATA LOADING TESTS
# ============================================================================

class TestDataLoading:
    """Test data loading functions."""
    
    def test_load_graph_from_edgelist(self, tmp_path, sample_graph):
        """Test loading graph from edge list file."""
        # Create test edge list file
        edgelist_path = tmp_path / "test_graph.txt"
        nx.write_edgelist(sample_graph, edgelist_path, data=False)
        
        # Load graph
        G_loaded = load_graph(str(edgelist_path))
        
        assert isinstance(G_loaded, nx.Graph)
        assert G_loaded.number_of_nodes() == sample_graph.number_of_nodes()
        assert G_loaded.number_of_edges() == sample_graph.number_of_edges()
    
    def test_save_and_load_graph(self, tmp_path, sample_graph):
        """Test saving and loading graph as pickle."""
        save_path = tmp_path / "test_graph.pkl"
        
        # Save graph
        save_graph(sample_graph, str(save_path))
        assert save_path.exists()
        
        # Load graph
        G_loaded = load_saved_graph(str(save_path))
        
        assert isinstance(G_loaded, nx.Graph)
        assert G_loaded.number_of_nodes() == sample_graph.number_of_nodes()
        assert G_loaded.number_of_edges() == sample_graph.number_of_edges()
    
    def test_load_communities(self, tmp_path, sample_communities):
        """Test loading communities from file."""
        # Create test communities file
        comm_path = tmp_path / "test_communities.txt"
        with open(comm_path, 'w') as f:
            for node, comm_id in sample_communities.items():
                f.write(f"{node} {comm_id}\n")
        
        # Load communities
        communities_loaded = load_communities(str(comm_path))
        
        assert isinstance(communities_loaded, dict)
        assert len(communities_loaded) == len(sample_communities)
    
    def test_train_test_split(self, sample_graph):
        """Test train/test split creation."""
        G_train, pos_test, neg_test = create_train_test_split(
            sample_graph, test_ratio=0.2, seed=42
        )
        
        assert isinstance(G_train, nx.Graph)
        assert isinstance(pos_test, list)
        assert isinstance(neg_test, list)
        assert len(pos_test) > 0
        assert len(neg_test) > 0
        assert len(pos_test) == len(neg_test)
        
        # Check that train graph has fewer edges
        assert G_train.number_of_edges() < sample_graph.number_of_edges()
    
    def test_remove_self_loops(self, sample_graph_with_selfloops):
        """Test self-loop removal."""
        num_selfloops_before = len(list(nx.selfloop_edges(sample_graph_with_selfloops)))
        assert num_selfloops_before > 0
        
        G_cleaned = remove_self_loops(sample_graph_with_selfloops)
        
        num_selfloops_after = len(list(nx.selfloop_edges(G_cleaned)))
        assert num_selfloops_after == 0
        assert isinstance(G_cleaned, nx.Graph)
    
    def test_get_largest_component(self, sample_graph):
        """Test largest component extraction."""
        # Add isolated nodes to create multiple components
        G = sample_graph.copy()
        G.add_node(1000)
        G.add_node(1001)
        
        G_largest = get_largest_component(G)
        
        assert isinstance(G_largest, nx.Graph)
        assert G_largest.number_of_nodes() <= G.number_of_nodes()
        assert nx.is_connected(G_largest) or G_largest.number_of_nodes() == 0
    
    def test_basic_statistics(self, sample_graph):
        """Test basic statistics computation."""
        stats = basic_statistics(sample_graph)
        
        assert isinstance(stats, dict)
        assert 'num_nodes' in stats
        assert 'num_edges' in stats
        assert 'density' in stats
        assert 'avg_degree' in stats
        assert 'num_components' in stats
        
        assert stats['num_nodes'] == sample_graph.number_of_nodes()
        assert stats['num_edges'] == sample_graph.number_of_edges()
        assert 0 <= stats['density'] <= 1


# ============================================================================
# CENTRALITY TESTS
# ============================================================================

class TestCentrality:
    """Test centrality algorithms."""
    
    def test_compute_pagerank(self, sample_graph):
        """Test PageRank computation."""
        pagerank = compute_pagerank(sample_graph)
        
        assert isinstance(pagerank, dict)
        assert len(pagerank) == sample_graph.number_of_nodes()
        
        # Check score ranges [0, 1]
        for node, score in pagerank.items():
            assert 0 <= score <= 1
            assert node in sample_graph.nodes()
        
        # Check scores sum to approximately 1
        total_score = sum(pagerank.values())
        assert abs(total_score - 1.0) < 0.01
    
    def test_compute_degree_centrality(self, sample_graph):
        """Test degree centrality computation."""
        degree_cent = compute_degree_centrality(sample_graph)
        
        assert isinstance(degree_cent, dict)
        assert len(degree_cent) == sample_graph.number_of_nodes()
        
        # Check score ranges [0, 1]
        for node, score in degree_cent.items():
            assert 0 <= score <= 1
            assert node in sample_graph.nodes()
    
    def test_compute_betweenness_centrality(self, sample_graph):
        """Test betweenness centrality computation."""
        betweenness = compute_betweenness_centrality(sample_graph, k=50)
        
        assert isinstance(betweenness, dict)
        # Note: k sampling may return fewer nodes
        assert len(betweenness) <= sample_graph.number_of_nodes()
        
        # Check score ranges [0, 1]
        for node, score in betweenness.items():
            assert 0 <= score <= 1
            assert node in sample_graph.nodes()
    
    def test_compute_hits(self, sample_graph):
        """Test HITS algorithm computation."""
        hubs, authorities = compute_hits(sample_graph)
        
        assert isinstance(hubs, dict)
        assert isinstance(authorities, dict)
        assert len(hubs) == sample_graph.number_of_nodes()
        assert len(authorities) == sample_graph.number_of_nodes()
        
        # Check score ranges [0, 1]
        for node in sample_graph.nodes():
            assert 0 <= hubs[node] <= 1
            assert 0 <= authorities[node] <= 1
    
    def test_get_top_k_nodes(self, sample_graph):
        """Test top-k nodes extraction."""
        pagerank = compute_pagerank(sample_graph)
        top_k = get_top_k_nodes(pagerank, k=10)
        
        assert isinstance(top_k, pd.DataFrame)
        assert len(top_k) == min(10, sample_graph.number_of_nodes())
        assert 'node_id' in top_k.columns
        assert 'score' in top_k.columns
        assert 'rank' in top_k.columns
        
        # Check scores are sorted descending
        scores = top_k['score'].values
        assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))


# ============================================================================
# COMMUNITY DETECTION TESTS
# ============================================================================

class TestCommunityDetection:
    """Test community detection algorithms."""
    
    def test_detect_communities_louvain(self, sample_graph):
        """Test Louvain community detection."""
        communities, modularity = detect_communities_louvain(sample_graph, seed=42)
        
        assert isinstance(communities, list)
        assert len(communities) > 0
        assert isinstance(modularity, float)
        assert -1 <= modularity <= 1
        
        # Check all nodes are in communities
        all_nodes = set()
        for comm in communities:
            assert isinstance(comm, set)
            all_nodes.update(comm)
        
        assert len(all_nodes) == sample_graph.number_of_nodes()
    
    def test_detect_communities_label_propagation(self, sample_graph):
        """Test Label Propagation community detection."""
        communities, modularity = detect_communities_label_propagation(sample_graph, seed=42)
        
        assert isinstance(communities, list)
        assert len(communities) > 0
        assert isinstance(modularity, float)
        assert -1 <= modularity <= 1
        
        # Check all nodes are in communities
        all_nodes = set()
        for comm in communities:
            assert isinstance(comm, set)
            all_nodes.update(comm)
        
        assert len(all_nodes) == sample_graph.number_of_nodes()
    
    def test_detect_communities_greedy_modularity(self, sample_graph):
        """Test Greedy Modularity community detection."""
        communities, modularity = detect_communities_greedy_modularity(sample_graph)
        
        assert isinstance(communities, list)
        assert len(communities) > 0
        assert isinstance(modularity, float)
        assert -1 <= modularity <= 1
    
    def test_calculate_modularity(self, sample_graph):
        """Test modularity calculation."""
        communities, _ = detect_communities_louvain(sample_graph, seed=42)
        modularity = calculate_modularity(sample_graph, communities)
        
        assert isinstance(modularity, float)
        assert -1 <= modularity <= 1
    
    def test_get_community_sizes(self, sample_graph):
        """Test community size extraction."""
        communities, _ = detect_communities_louvain(sample_graph, seed=42)
        sizes = get_community_sizes(communities)
        
        assert isinstance(sizes, pd.Series)
        assert len(sizes) == len(communities)
        assert all(size > 0 for size in sizes)
        assert sizes.sum() == sample_graph.number_of_nodes()


# ============================================================================
# LINK PREDICTION TESTS
# ============================================================================

class TestLinkPrediction:
    """Test link prediction functions."""
    
    def test_common_neighbors_score(self, sample_graph, sample_edge_list):
        """Test Common Neighbors score computation."""
        scores = common_neighbors_score(sample_graph, sample_edge_list)
        
        assert isinstance(scores, list)
        assert len(scores) == len(sample_edge_list)
        assert all(isinstance(score, (int, float)) for score in scores)
        assert all(score >= 0 for score in scores)
    
    def test_adamic_adar_score(self, sample_graph, sample_edge_list):
        """Test Adamic-Adar score computation."""
        scores = adamic_adar_score(sample_graph, sample_edge_list)
        
        assert isinstance(scores, list)
        assert len(scores) == len(sample_edge_list)
        assert all(isinstance(score, (int, float)) for score in scores)
        assert all(score >= 0 for score in scores)
    
    def test_jaccard_coefficient_score(self, sample_graph, sample_edge_list):
        """Test Jaccard Coefficient score computation."""
        scores = jaccard_coefficient_score(sample_graph, sample_edge_list)
        
        assert isinstance(scores, list)
        assert len(scores) == len(sample_edge_list)
        assert all(isinstance(score, (int, float)) for score in scores)
        assert all(0 <= score <= 1 for score in scores)
    
    def test_evaluate_link_prediction(self, sample_graph):
        """Test link prediction evaluation."""
        # Create train/test split
        G_train, pos_test, neg_test = create_train_test_split(
            sample_graph, test_ratio=0.2, seed=42
        )
        
        # Use small sample for testing
        pos_sample = pos_test[:min(10, len(pos_test))]
        neg_sample = neg_test[:min(10, len(neg_test))]
        
        metrics = evaluate_link_prediction(
            G_train, pos_sample, neg_sample, method='adamic_adar'
        )
        
        assert isinstance(metrics, dict)
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'auc_roc' in metrics
        assert 'auc_pr' in metrics
        
        # Check metric ranges
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1'] <= 1
        assert 0 <= metrics['auc_roc'] <= 1 or np.isnan(metrics['auc_roc'])
        assert 0 <= metrics['auc_pr'] <= 1 or np.isnan(metrics['auc_pr'])


# ============================================================================
# ML LINK PREDICTION TESTS
# ============================================================================

class TestMLLinkPrediction:
    """Test ML-based link prediction functions."""
    
    def test_extract_edge_features(self, sample_graph, sample_edge_list):
        """Test edge feature extraction."""
        features_df = extract_edge_features(sample_graph, sample_edge_list)
        
        assert isinstance(features_df, pd.DataFrame)
        assert len(features_df) == len(sample_edge_list)
        assert 'node1' in features_df.columns
        assert 'node2' in features_df.columns
        
        # Check feature columns
        expected_features = [
            'common_neighbors', 'jaccard_coefficient', 'adamic_adar',
            'degree_u', 'degree_v', 'degree_product', 'degree_sum',
            'clustering_u', 'clustering_v'
        ]
        for feat in expected_features:
            assert feat in features_df.columns
    
    def test_prepare_training_data(self, sample_graph):
        """Test training data preparation."""
        # Create sample edges
        edges = list(sample_graph.edges())
        pos_edges = edges[:min(20, len(edges))]
        neg_edges = [(1000, 1001), (1002, 1003), (1004, 1005)]
        
        X_train, y_train = prepare_training_data(
            sample_graph, pos_edges, neg_edges
        )
        
        assert isinstance(X_train, np.ndarray)
        assert isinstance(y_train, np.ndarray)
        assert X_train.shape[0] == len(pos_edges) + len(neg_edges)
        assert len(y_train) == len(pos_edges) + len(neg_edges)
        assert X_train.shape[1] == 9  # 9 features
        assert set(y_train) == {0, 1}
        assert np.sum(y_train == 1) == len(pos_edges)
        assert np.sum(y_train == 0) == len(neg_edges)
    
    def test_train_random_forest(self, sample_graph):
        """Test Random Forest model training."""
        # Prepare training data
        edges = list(sample_graph.edges())
        pos_edges = edges[:min(20, len(edges))]
        neg_edges = [(1000, 1001), (1002, 1003)]
        
        X_train, y_train = prepare_training_data(
            sample_graph, pos_edges, neg_edges
        )
        
        # Train model
        model, scaler = train_random_forest(
            X_train, y_train, n_estimators=10, random_state=42
        )
        
        assert model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
    
    def test_evaluate_ml_model(self, sample_graph):
        """Test ML model evaluation."""
        # Prepare training data
        edges = list(sample_graph.edges())
        pos_edges = edges[:min(15, len(edges))]
        neg_edges = [(1000, 1001), (1002, 1003), (1004, 1005)]
        
        X_train, y_train = prepare_training_data(
            sample_graph, pos_edges, neg_edges
        )
        
        # Train model
        model, scaler = train_random_forest(
            X_train, y_train, n_estimators=10, random_state=42
        )
        
        # Evaluate
        pos_test = edges[min(15, len(edges)):min(20, len(edges))]
        neg_test = [(2000, 2001), (2002, 2003)]
        
        metrics, predictions = evaluate_ml_model(
            model, sample_graph, pos_test, neg_test, scaler=scaler
        )
        
        assert isinstance(metrics, dict)
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'auc_roc' in metrics
        assert isinstance(predictions, pd.DataFrame)
        assert 'probability' in predictions.columns
        assert 'prediction' in predictions.columns


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for full workflows."""
    
    def test_full_centrality_workflow(self, sample_graph):
        """Test complete centrality analysis workflow."""
        # Compute all centralities
        pagerank = compute_pagerank(sample_graph)
        degree_cent = compute_degree_centrality(sample_graph)
        betweenness = compute_betweenness_centrality(sample_graph, k=50)
        
        # Get top-k for each
        top_pr = get_top_k_nodes(pagerank, k=10)
        top_deg = get_top_k_nodes(degree_cent, k=10)
        top_bt = get_top_k_nodes(betweenness, k=10)
        
        assert len(top_pr) == min(10, sample_graph.number_of_nodes())
        assert len(top_deg) == min(10, sample_graph.number_of_nodes())
        assert len(top_bt) == min(10, len(betweenness))
    
    def test_full_community_workflow(self, sample_graph):
        """Test complete community detection workflow."""
        # Detect communities
        communities, modularity = detect_communities_louvain(sample_graph, seed=42)
        
        # Calculate modularity
        mod = calculate_modularity(sample_graph, communities)
        
        # Get sizes
        sizes = get_community_sizes(communities)
        
        assert abs(mod - modularity) < 0.01  # Should match
        assert len(sizes) == len(communities)
        assert sizes.sum() == sample_graph.number_of_nodes()
    
    def test_full_link_prediction_workflow(self, sample_graph):
        """Test complete link prediction workflow."""
        # Create train/test split
        G_train, pos_test, neg_test = create_train_test_split(
            sample_graph, test_ratio=0.2, seed=42
        )
        
        # Evaluate similarity method
        pos_sample = pos_test[:min(10, len(pos_test))]
        neg_sample = neg_test[:min(10, len(neg_test))]
        
        metrics = evaluate_link_prediction(
            G_train, pos_sample, neg_sample, method='adamic_adar'
        )
        
        # Train ML model
        train_pos = pos_test[:min(15, len(pos_test))]
        train_neg = neg_test[:min(15, len(neg_test))]
        
        X_train, y_train = prepare_training_data(G_train, train_pos, train_neg)
        model, scaler = train_random_forest(X_train, y_train, n_estimators=10, random_state=42)
        
        ml_metrics, _ = evaluate_ml_model(
            model, G_train, pos_sample, neg_sample, scaler=scaler
        )
        
        assert isinstance(metrics, dict)
        assert isinstance(ml_metrics, dict)
        assert 'f1' in metrics
        assert 'f1' in ml_metrics


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_graph(self):
        """Test handling of empty graph."""
        G = nx.Graph()
        
        # Should not crash
        pagerank = compute_pagerank(G)
        assert len(pagerank) == 0
    
    def test_single_node_graph(self):
        """Test handling of single node graph."""
        G = nx.Graph()
        G.add_node(0)
        
        pagerank = compute_pagerank(G)
        assert len(pagerank) == 1
        assert pagerank[0] == 1.0
    
    def test_disconnected_graph(self, sample_graph):
        """Test handling of disconnected graph."""
        # Add isolated nodes
        G = sample_graph.copy()
        G.add_node(1000)
        G.add_node(1001)
        
        # Should still work
        pagerank = compute_pagerank(G)
        assert len(pagerank) == G.number_of_nodes()
    
    def test_empty_edge_list(self, sample_graph):
        """Test handling of empty edge list."""
        scores = common_neighbors_score(sample_graph, [])
        assert len(scores) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

