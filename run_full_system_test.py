"""
Comprehensive system test - runs entire pipeline end-to-end.

This script verifies that all modules work correctly by:
1. Testing data loading and preprocessing
2. Running all analysis modules
3. Executing notebook code cells
4. Verifying output generation
"""

import os
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Color output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_section(title):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")

def print_success(msg):
    print(f"{Colors.GREEN}✅ {msg}{Colors.END}")

def print_error(msg):
    print(f"{Colors.RED}❌ {msg}{Colors.END}")

def print_warning(msg):
    print(f"{Colors.YELLOW}⚠️  {msg}{Colors.END}")

def print_info(msg):
    print(f"{Colors.BLUE}ℹ️  {msg}{Colors.END}")

# Track results
results = {
    'passed': [],
    'failed': [],
    'warnings': []
}

def test_module(name, test_func):
    """Run a test and track results."""
    try:
        print_info(f"Testing {name}...")
        start_time = time.time()
        test_func()
        elapsed = time.time() - start_time
        print_success(f"{name} passed ({elapsed:.2f}s)")
        results['passed'].append(name)
        return True
    except Exception as e:
        print_error(f"{name} failed: {str(e)}")
        traceback.print_exc()
        results['failed'].append((name, str(e)))
        return False

# ============================================================================
# TEST 1: Data Loading and Preprocessing
# ============================================================================
def test_data_loading():
    print_section("TEST 1: Data Loading and Preprocessing")
    
    import networkx as nx
    from data_loader import load_graph, save_graph, load_saved_graph
    from preprocessing import remove_self_loops, get_largest_component, create_train_test_split
    
    # Create sample graph if needed
    if not os.path.exists('data/processed/graph.pkl'):
        print_warning("Preprocessed graph not found, creating sample...")
        G = nx.erdos_renyi_graph(1000, 0.01, seed=42)
        os.makedirs('data/processed', exist_ok=True)
        save_graph(G, 'data/processed/graph.pkl')
    
    # Load graph
    G = load_saved_graph('data/processed/graph.pkl')
    assert len(G) > 0, "Graph is empty"
    
    # Preprocessing
    G_clean = remove_self_loops(G.copy())
    G_lcc = get_largest_component(G_clean)
    assert len(G_lcc) > 0, "Largest component is empty"
    
    # Train/test split
    G_train, pos_test, neg_test = create_train_test_split(G_lcc, test_ratio=0.2, seed=42)
    assert len(pos_test) > 0, "No positive test edges"
    assert len(neg_test) > 0, "No negative test edges"
    
    print_success(f"Graph: {len(G_lcc)} nodes, {G_lcc.number_of_edges()} edges")
    print_success(f"Train/test split: {len(pos_test)} pos, {len(neg_test)} neg")

# ============================================================================
# TEST 2: Centrality Analysis
# ============================================================================
def test_centrality():
    print_section("TEST 2: Centrality Analysis")
    
    from centrality_analysis import (
        compute_pagerank, compute_degree_centrality,
        compute_betweenness_centrality, compute_hits
    )
    import networkx as nx
    
    G = nx.erdos_renyi_graph(500, 0.02, seed=42)
    
    pr = compute_pagerank(G)
    assert len(pr) == len(G), "PageRank missing nodes"
    
    dc = compute_degree_centrality(G)
    assert len(dc) == len(G), "Degree centrality missing nodes"
    
    bc = compute_betweenness_centrality(G, k=100)
    assert len(bc) > 0, "Betweenness centrality empty"
    
    hubs, auth = compute_hits(G)
    assert len(hubs) == len(G), "HITS missing nodes"
    
    print_success("All centrality measures computed")

# ============================================================================
# TEST 3: Community Detection
# ============================================================================
def test_community_detection():
    print_section("TEST 3: Community Detection")
    
    from community_detection import (
        detect_communities_louvain,
        detect_communities_label_propagation,
        detect_communities_greedy_modularity
    )
    import networkx as nx
    
    G = nx.erdos_renyi_graph(500, 0.02, seed=42)
    
    communities_louvain = detect_communities_louvain(G)
    assert len(communities_louvain) > 0, "Louvain found no communities"
    
    communities_lp = detect_communities_label_propagation(G)
    assert len(communities_lp) > 0, "Label propagation found no communities"
    
    communities_gm = detect_communities_greedy_modularity(G)
    assert len(communities_gm) > 0, "Greedy modularity found no communities"
    
    print_success(f"Louvain: {len(communities_louvain)} communities")
    print_success(f"Label Propagation: {len(communities_lp)} communities")
    print_success(f"Greedy Modularity: {len(communities_gm)} communities")

# ============================================================================
# TEST 4: Link Prediction
# ============================================================================
def test_link_prediction():
    print_section("TEST 4: Link Prediction")
    
    from link_prediction import (
        common_neighbors_score, adamic_adar_score, jaccard_coefficient_score
    )
    from ml_link_prediction import extract_edge_features, train_random_forest
    import networkx as nx
    import numpy as np
    
    G = nx.erdos_renyi_graph(500, 0.02, seed=42)
    
    # Create test edges
    test_edges = list(G.edges())[:100]
    
    # Similarity scores
    cn_scores = common_neighbors_score(G, test_edges)
    assert len(cn_scores) == len(test_edges), "Common neighbors scores mismatch"
    
    aa_scores = adamic_adar_score(G, test_edges)
    assert len(aa_scores) == len(test_edges), "Adamic-Adar scores mismatch"
    
    jc_scores = jaccard_coefficient_score(G, test_edges)
    assert len(jc_scores) == len(test_edges), "Jaccard scores mismatch"
    
    # ML features
    features = extract_edge_features(G, test_edges)
    assert features.shape[0] == len(test_edges), "Feature extraction mismatch"
    
    print_success("Link prediction methods working")

# ============================================================================
# TEST 5: Spam Farm Generation
# ============================================================================
def test_spam_farms():
    print_section("TEST 5: Spam Farm Generation")
    
    from spam_farm_generator import create_simple_spam_farm
    import networkx as nx
    
    G = nx.erdos_renyi_graph(500, 0.02, seed=42)
    
    G_spam, spam_nodes, target = create_simple_spam_farm(
        G.copy(), None, m=100, external_links=5
    )
    
    assert len(G_spam) > len(G), "Spam farm didn't add nodes"
    assert len(spam_nodes) == 100, "Wrong number of spam nodes"
    assert target in G_spam, "Target not in graph"
    
    print_success(f"Spam farm: {len(spam_nodes)} supporting pages, target={target}")

# ============================================================================
# TEST 6: TrustRank
# ============================================================================
def test_trustrank():
    print_section("TEST 6: TrustRank")
    
    from trustrank import select_trusted_pages, compute_trustrank
    import networkx as nx
    
    G = nx.erdos_renyi_graph(500, 0.02, seed=42)
    pr = nx.pagerank(G, alpha=0.85)
    
    trusted = select_trusted_pages(G, method='top_pagerank', k=50, pagerank_scores=pr)
    assert len(trusted) == 50, "Wrong number of trusted pages"
    
    tr_scores, _ = compute_trustrank(G, trusted, beta=0.85, max_iter=50)
    assert len(tr_scores) == len(G), "TrustRank missing nodes"
    
    print_success(f"TrustRank computed with {len(trusted)} trusted pages")

# ============================================================================
# TEST 7: Spam Mass
# ============================================================================
def test_spam_mass():
    print_section("TEST 7: Spam Mass")
    
    from spam_mass import calculate_spam_mass, classify_by_spam_mass
    import networkx as nx
    
    G = nx.erdos_renyi_graph(500, 0.02, seed=42)
    pr = nx.pagerank(G, alpha=0.85)
    
    # Create fake TrustRank (just use PR for testing)
    tr = {n: pr[n] * 0.9 for n in G.nodes()}
    
    spam_mass = calculate_spam_mass(pr, tr)
    assert len(spam_mass) == len(G), "Spam mass missing nodes"
    
    classifications = classify_by_spam_mass(spam_mass, threshold=0.7)
    assert len(classifications) == len(G), "Classifications missing nodes"
    
    print_success("Spam mass calculated and classified")

# ============================================================================
# TEST 8: Structural Detection
# ============================================================================
def test_structural_detection():
    print_section("TEST 8: Structural Detection")
    
    from structural_spam_detection import (
        detect_star_patterns,
        detect_honeypot_patterns,
        ensemble_structural_detection
    )
    import networkx as nx
    
    G = nx.erdos_renyi_graph(500, 0.02, seed=42)
    
    stars = detect_star_patterns(G, min_supporters=10)
    assert isinstance(stars, list), "Star patterns not a list"
    
    honeypots = detect_honeypot_patterns(G, min_in_degree=5, max_out_degree=3)
    assert isinstance(honeypots, list), "Honeypot patterns not a list"
    
    ensemble = ensemble_structural_detection(G, methods=['star', 'honeypot'])
    assert 'node_id' in ensemble.columns, "Ensemble results missing columns"
    
    print_success("Structural detection methods working")

# ============================================================================
# TEST 9: Spam Analysis Pipeline
# ============================================================================
def test_spam_pipeline():
    print_section("TEST 9: Spam Analysis Pipeline")
    
    from spam_analysis_pipeline import run_complete_spam_experiment
    import networkx as nx
    
    G = nx.erdos_renyi_graph(500, 0.02, seed=42)
    
    config = {
        'spam_type': 'simple',
        'm': 50,
        'external_links': 5,
        'beta': 0.85,
        'detection_methods': ['trustrank', 'spam_mass'],
        'trusted_nodes': None
    }
    
    results = run_complete_spam_experiment(
        G, config, save_intermediate=False, output_dir=None
    )
    
    assert results.experiment_id is not None, "Experiment ID missing"
    assert results.spam_metadata is not None, "Spam metadata missing"
    
    print_success(f"Pipeline experiment completed: {results.experiment_id}")

# ============================================================================
# TEST 10: Effectiveness Analysis
# ============================================================================
def test_effectiveness():
    print_section("TEST 10: Effectiveness Analysis")
    
    from spam_effectiveness_analysis import (
        calculate_pagerank_amplification,
        analyze_spam_farm_roi,
        analyze_network_damage
    )
    from spam_farm_generator import create_simple_spam_farm
    import networkx as nx
    
    G_original = nx.erdos_renyi_graph(500, 0.02, seed=42)
    # Use existing node as target so it exists in original graph
    target = list(G_original.nodes())[0]
    G_spam, spam_nodes, target = create_simple_spam_farm(
        G_original.copy(), target, m=50, external_links=5
    )
    
    spam_metadata = {target: {'m': 50, 'spam_nodes': spam_nodes, 'farm_type': 'simple'}}
    
    # Amplification
    amp_df = calculate_pagerank_amplification(
        G_original, G_spam, [target], spam_metadata
    )
    assert len(amp_df) > 0, "Amplification analysis empty"
    
    # ROI
    pr_orig = nx.pagerank(G_original, alpha=0.85)
    pr_spam = nx.pagerank(G_spam, alpha=0.85)
    boosts = {target: pr_spam[target] - pr_orig.get(target, 0)}
    roi_df = analyze_spam_farm_roi(spam_metadata, boosts)
    assert len(roi_df) > 0, "ROI analysis empty"
    
    # Network damage
    damage = analyze_network_damage(G_original, G_spam)
    assert 'spam_nodes_added' in damage, "Network damage missing fields"
    
    print_success("Effectiveness analysis completed")

# ============================================================================
# TEST 11: Notebook Execution (Simulated)
# ============================================================================
def test_notebook_code():
    print_section("TEST 11: Notebook Code Execution")
    
    # Execute key notebook code cells
    import networkx as nx
    import pandas as pd
    import numpy as np
    
    # Simulate notebook imports
    sys.path.insert(0, 'src')
    
    # Test data loading code
    from data_loader import load_saved_graph
    if os.path.exists('data/processed/graph.pkl'):
        G = load_saved_graph('data/processed/graph.pkl')
        print_success("Notebook data loading works")
    else:
        print_warning("Preprocessed graph not found, skipping notebook data test")
    
    # Test spam farm creation code
    from spam_farm_generator import create_simple_spam_farm
    G_test = nx.erdos_renyi_graph(200, 0.03, seed=42)
    G_spam, spam_nodes, target = create_simple_spam_farm(
        G_test.copy(), None, m=50, external_links=5
    )
    print_success("Notebook spam farm creation works")
    
    # Test visualization code (just check imports)
    import matplotlib.pyplot as plt
    import seaborn as sns
    print_success("Notebook visualization imports work")

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("="*60)
    print("COMPREHENSIVE SYSTEM TEST")
    print("="*60)
    print(f"{Colors.END}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    start_time = time.time()
    
    # Run all tests
    tests = [
        ("Data Loading", test_data_loading),
        ("Centrality Analysis", test_centrality),
        ("Community Detection", test_community_detection),
        ("Link Prediction", test_link_prediction),
        ("Spam Farm Generation", test_spam_farms),
        ("TrustRank", test_trustrank),
        ("Spam Mass", test_spam_mass),
        ("Structural Detection", test_structural_detection),
        ("Spam Pipeline", test_spam_pipeline),
        ("Effectiveness Analysis", test_effectiveness),
        ("Notebook Code", test_notebook_code),
    ]
    
    for name, test_func in tests:
        test_module(name, test_func)
    
    elapsed = time.time() - start_time
    
    # Print summary
    print_section("TEST SUMMARY")
    
    print(f"{Colors.GREEN}Passed: {len(results['passed'])}{Colors.END}")
    for name in results['passed']:
        print(f"  ✅ {name}")
    
    if results['failed']:
        print(f"\n{Colors.RED}Failed: {len(results['failed'])}{Colors.END}")
        for name, error in results['failed']:
            print(f"  ❌ {name}: {error}")
    
    print(f"\n{Colors.BLUE}Total time: {elapsed:.2f} seconds{Colors.END}")
    
    # Exit code
    if results['failed']:
        print(f"\n{Colors.RED}❌ SOME TESTS FAILED{Colors.END}")
        sys.exit(1)
    else:
        print(f"\n{Colors.GREEN}✅ ALL TESTS PASSED{Colors.END}")
        sys.exit(0)

if __name__ == "__main__":
    main()

