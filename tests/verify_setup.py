#!/usr/bin/env python3
"""
Verification script for Amazon Co-Purchasing Network project setup.

This script verifies that all required files exist and that all modules
work correctly before proceeding to Module 2.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))
os.chdir(project_root)

# Try to import colorama for colored output
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    HAS_COLORAMA = True
except ImportError:
    HAS_COLORAMA = False
    # Define dummy colors if colorama not available
    class Fore:
        GREEN = ''
        RED = ''
        YELLOW = ''
        CYAN = ''
        RESET = ''
    class Style:
        BRIGHT = ''
        RESET_ALL = ''

def print_success(message):
    """Print success message in green."""
    if HAS_COLORAMA:
        print(f"{Fore.GREEN}✅ {message}{Style.RESET_ALL}")
    else:
        print(f"✅ {message}")

def print_error(message):
    """Print error message in red."""
    if HAS_COLORAMA:
        print(f"{Fore.RED}❌ {message}{Style.RESET_ALL}")
    else:
        print(f"❌ {message}")

def print_info(message):
    """Print info message."""
    if HAS_COLORAMA:
        print(f"{Fore.CYAN}   {message}{Style.RESET_ALL}")
    else:
        print(f"   {message}")

def print_header(message):
    """Print section header."""
    if HAS_COLORAMA:
        print(f"\n{Fore.CYAN}{Style.BRIGHT}{message}{Style.RESET_ALL}")
    else:
        print(f"\n{message}")

def test_1_check_files():
    """Test 1: Check if all required files exist."""
    print_header("[1/6] Checking required files...")
    
    required_files = [
        "data/raw/com-amazon.ungraph.txt.gz",
        "data/raw/com-amazon.all.cmty.txt.gz",
        "src/data_loader.py",
        "src/preprocessing.py",
        "src/exploratory_analysis.py",
    ]
    
    all_found = True
    for filepath in required_files:
        if os.path.exists(filepath):
            print_success(f"Found: {filepath}")
        else:
            print_error(f"Missing: {filepath}")
            all_found = False
    
    return all_found

def test_2_data_loading():
    """Test 2: Test data loading functions."""
    print_header("[2/6] Testing data loading...")
    
    try:
        from data_loader import load_graph, load_communities
        
        # Load graph
        graph_path = "data/raw/com-amazon.ungraph.txt.gz"
        if not os.path.exists(graph_path):
            print_error(f"Graph file not found: {graph_path}")
            return False
        
        print_info("Loading graph...")
        G = load_graph(graph_path, is_gzipped=True)
        
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        is_undirected = not G.is_directed()
        
        print_success("Graph loaded successfully")
        print_info(f"   - Nodes: {num_nodes:,}")
        print_info(f"   - Edges: {num_edges:,}")
        print_info(f"   - Type: {'Undirected' if is_undirected else 'Directed'}")
        
        if not is_undirected:
            print_error("Graph should be undirected")
            return False
        
        # Load communities
        communities_path = "data/raw/com-amazon.all.cmty.txt.gz"
        if not os.path.exists(communities_path):
            print_error(f"Communities file not found: {communities_path}")
            return False
        
        print_info("Loading communities...")
        communities = load_communities(communities_path, is_gzipped=True)
        
        num_communities = len(communities)
        print_success("Communities loaded successfully")
        print_info(f"   - Number of communities: {num_communities:,}")
        
        return True, G, communities
    
    except Exception as e:
        print_error(f"Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

def test_3_preprocessing(G):
    """Test 3: Test preprocessing functions."""
    print_header("[3/6] Testing preprocessing...")
    
    if G is None:
        print_error("Cannot test preprocessing: graph not loaded")
        return False
    
    try:
        from preprocessing import (
            remove_self_loops,
            get_largest_component,
            basic_statistics
        )
        
        # Test remove_self_loops
        print_info("Removing self-loops...")
        original_nodes = G.number_of_nodes()
        original_edges = G.number_of_edges()
        
        G_cleaned = remove_self_loops(G)
        self_loops_removed = original_edges - G_cleaned.number_of_edges()
        
        print_success(f"Self-loops removed: {self_loops_removed} loops found")
        
        # Test get_largest_component
        print_info("Extracting largest component...")
        G_largest = get_largest_component(G_cleaned)
        
        largest_nodes = G_largest.number_of_nodes()
        largest_edges = G_largest.number_of_edges()
        
        print_success("Largest component extracted")
        print_info(f"   - Original nodes: {original_nodes:,}")
        print_info(f"   - Largest component nodes: {largest_nodes:,}")
        print_info(f"   - Original edges: {original_edges:,}")
        print_info(f"   - Largest component edges: {largest_edges:,}")
        
        # Test basic_statistics
        print_info("Computing basic statistics...")
        stats = basic_statistics(G_largest)
        
        print_success("Basic statistics computed")
        print_info(f"   - Density: {stats['density']:.8f}")
        print_info(f"   - Average degree: {stats['avg_degree']:.4f}")
        print_info(f"   - Connected components: {stats['num_components']}")
        
        return True, G_largest
    
    except Exception as e:
        print_error(f"Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_4_train_test_split(G_largest):
    """Test 4: Test train/test split creation."""
    print_header("[4/6] Testing train/test split...")
    
    if G_largest is None:
        print_error("Cannot test train/test split: graph not loaded")
        return False
    
    try:
        from preprocessing import create_train_test_split
        
        print_info("Creating train/test split (80/20)...")
        G_train, pos_test, neg_test = create_train_test_split(
            G_largest, test_ratio=0.2, seed=42
        )
        
        train_edges = G_train.number_of_edges()
        num_pos_test = len(pos_test)
        num_neg_test = len(neg_test)
        
        # Verify split ratio
        total_original_edges = train_edges + num_pos_test
        actual_ratio = num_pos_test / total_original_edges if total_original_edges > 0 else 0
        
        print_success("Train/test split created")
        print_info(f"   - Training edges: {train_edges:,}")
        print_info(f"   - Test edges (positive): {num_pos_test:,}")
        print_info(f"   - Test edges (negative): {num_neg_test:,}")
        print_info(f"   - Split ratio: {actual_ratio*100:.1f}% / {(1-actual_ratio)*100:.1f}%")
        
        # Verify sizes match
        if num_pos_test != num_neg_test:
            print_error(f"Positive and negative test edges don't match: {num_pos_test} vs {num_neg_test}")
            return False
        
        if abs(actual_ratio - 0.2) > 0.01:
            print_error(f"Split ratio incorrect: expected ~20%, got {actual_ratio*100:.1f}%")
            return False
        
        return True
    
    except Exception as e:
        print_error(f"Train/test split failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_5_exploratory_analysis(G_largest):
    """Test 5: Test exploratory analysis functions."""
    print_header("[5/6] Testing exploratory analysis...")
    
    if G_largest is None:
        print_error("Cannot test exploratory analysis: graph not loaded")
        return False
    
    try:
        from exploratory_analysis import (
            degree_distribution,
            compute_network_stats
        )
        
        # Test degree distribution
        print_info("Computing degree distribution...")
        degree_series = degree_distribution(G_largest)
        
        print_success("Degree distribution computed")
        print_info(f"   - Mean degree: {degree_series.mean():.2f}")
        print_info(f"   - Median degree: {degree_series.median():.2f}")
        print_info(f"   - Max degree: {degree_series.max()}")
        
        # Test network statistics (with sampling for speed)
        print_info("Computing network statistics (this may take a while)...")
        network_stats = compute_network_stats(G_largest, sample_size=5000)
        
        print_success("Network statistics generated")
        print_info(f"   - Average clustering: {network_stats['avg_clustering']:.6f}")
        print_info(f"   - Number of triangles: {network_stats['num_triangles']:,}")
        print_info(f"   - Degree assortativity: {network_stats['degree_assortativity']:.6f}")
        
        return True
    
    except Exception as e:
        print_error(f"Exploratory analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all verification tests."""
    print("=" * 60)
    print("🔍 VERIFICATION SCRIPT")
    print("=" * 60)
    
    all_passed = True
    G = None
    communities = None
    G_largest = None
    
    # Test 1: Check files
    if not test_1_check_files():
        all_passed = False
        print_error("Test 1 failed: Required files missing")
        return 1
    
    # Test 2: Data loading
    result = test_2_data_loading()
    if isinstance(result, tuple):
        if result[0]:
            G = result[1]
            communities = result[2]
        else:
            all_passed = False
            return 1
    else:
        all_passed = False
        return 1
    
    # Test 3: Preprocessing
    result = test_3_preprocessing(G)
    if isinstance(result, tuple):
        if result[0]:
            G_largest = result[1]
        else:
            all_passed = False
            return 1
    else:
        all_passed = False
        return 1
    
    # Test 4: Train/test split
    if not test_4_train_test_split(G_largest):
        all_passed = False
        return 1
    
    # Test 5: Exploratory analysis
    if not test_5_exploratory_analysis(G_largest):
        all_passed = False
        return 1
    
    # Final validation
    print_header("[6/6] Final validation...")
    print("=" * 60)
    
    if all_passed:
        print_success("ALL TESTS PASSED!")
        print("=" * 60)
        print()
        print("You can now proceed to Module 2 (Centrality Analysis)")
        return 0
    else:
        print_error("SOME TESTS FAILED!")
        print("=" * 60)
        print()
        print("Please fix the errors above before proceeding.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

