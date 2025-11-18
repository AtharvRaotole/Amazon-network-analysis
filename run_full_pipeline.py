#!/usr/bin/env python3
"""
Full Pipeline Execution Script

This script runs the complete data processing pipeline:
1. Load Amazon dataset
2. Preprocess (remove self-loops, extract largest component)
3. Create train/test splits
4. Generate statistics and visualizations
5. Save all outputs
"""

import os
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))
os.chdir(project_root)

from data_loader import (
    load_graph, 
    load_communities,
    save_graph
)
from preprocessing import (
    remove_self_loops,
    get_largest_component,
    basic_statistics,
    create_train_test_split,
    save_splits
)
from exploratory_analysis import (
    degree_distribution,
    plot_degree_distribution,
    compute_network_stats,
    generate_statistics_report
)

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def main():
    """Run the complete pipeline."""
    print("=" * 70)
    print("  AMAZON CO-PURCHASING NETWORK - FULL PIPELINE EXECUTION")
    print("=" * 70)
    
    # Ensure directories exist
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/processed/splits", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)
    
    try:
        # ====================================================================
        # STEP 1: LOAD DATASET
        # ====================================================================
        print_section("STEP 1: Loading Dataset")
        
        graph_path = "data/raw/com-amazon.ungraph.txt.gz"
        communities_path = "data/raw/com-amazon.all.cmty.txt.gz"
        
        print(f"Loading graph from: {graph_path}")
        G = load_graph(graph_path, is_gzipped=True)
        print(f"✅ Graph loaded: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
        
        print(f"\nLoading communities from: {communities_path}")
        communities = load_communities(communities_path, is_gzipped=True)
        print(f"✅ Communities loaded: {len(communities):,} communities")
        
        # ====================================================================
        # STEP 2: PREPROCESSING
        # ====================================================================
        print_section("STEP 2: Preprocessing")
        
        print("Removing self-loops...")
        G_cleaned = remove_self_loops(G)
        print(f"✅ Self-loops removed")
        
        print("\nExtracting largest connected component...")
        G_largest = get_largest_component(G_cleaned)
        print(f"✅ Largest component: {G_largest.number_of_nodes():,} nodes, "
              f"{G_largest.number_of_edges():,} edges")
        
        node_retention = (G_largest.number_of_nodes() / G.number_of_nodes()) * 100
        print(f"   Node retention: {node_retention:.2f}%")
        
        # Compute basic statistics
        print("\nComputing basic statistics...")
        stats = basic_statistics(G_largest)
        print("✅ Basic statistics computed:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.6f}")
            else:
                print(f"   {key}: {value:,}")
        
        # ====================================================================
        # STEP 3: SAVE CLEANED GRAPH
        # ====================================================================
        print_section("STEP 3: Saving Cleaned Graph")
        
        cleaned_graph_path = "data/processed/amazon_graph_cleaned.pkl"
        save_graph(G_largest, cleaned_graph_path)
        print(f"✅ Cleaned graph saved to: {cleaned_graph_path}")
        
        # ====================================================================
        # STEP 4: CREATE TRAIN/TEST SPLIT
        # ====================================================================
        print_section("STEP 4: Creating Train/Test Split")
        
        print("Creating 80/20 train/test split for link prediction...")
        G_train, positive_test_edges, negative_test_edges = create_train_test_split(
            G_largest, test_ratio=0.2, seed=42
        )
        
        print("✅ Train/test split created:")
        print(f"   Training graph: {G_train.number_of_nodes():,} nodes, "
              f"{G_train.number_of_edges():,} edges")
        print(f"   Positive test edges: {len(positive_test_edges):,}")
        print(f"   Negative test edges: {len(negative_test_edges):,}")
        
        # Save splits
        print("\nSaving train/test splits...")
        save_splits(G_train, positive_test_edges, negative_test_edges, 
                   output_dir="data/processed/splits")
        print("✅ Splits saved to: data/processed/splits/")
        
        # ====================================================================
        # STEP 5: EXPLORATORY ANALYSIS
        # ====================================================================
        print_section("STEP 5: Exploratory Analysis")
        
        # Degree distribution
        print("Computing degree distribution...")
        degree_series = degree_distribution(G_largest)
        print("✅ Degree distribution computed")
        print(f"   Mean degree: {degree_series.mean():.2f}")
        print(f"   Median degree: {degree_series.median():.2f}")
        print(f"   Max degree: {degree_series.max()}")
        
        # Plot degree distribution
        print("\nCreating degree distribution plots...")
        plot_degree_distribution(G_largest, save_path="results/figures/degree_distribution.png")
        print("✅ Saved: results/figures/degree_distribution.png")
        
        # Network statistics
        print("\nComputing comprehensive network statistics...")
        print("   (This may take a few minutes for large graphs)")
        network_stats = compute_network_stats(G_largest, sample_size=10000)
        print("✅ Network statistics computed:")
        print(f"   Average clustering: {network_stats['avg_clustering']:.6f}")
        print(f"   Number of triangles: {network_stats['num_triangles']:,}")
        print(f"   Degree assortativity: {network_stats['degree_assortativity']:.6f}")
        
        # Generate statistics report
        print("\nGenerating statistics report...")
        generate_statistics_report(G_largest, communities=communities, 
                                 output_path="results/tables/network_statistics")
        print("✅ Statistics report saved:")
        print("   - results/tables/network_statistics.csv")
        print("   - results/tables/network_statistics.txt")
        
        # ====================================================================
        # STEP 6: COMMUNITY ANALYSIS
        # ====================================================================
        print_section("STEP 6: Community Analysis")
        
        community_sizes = [len(nodes) for nodes in communities.values()]
        unique_nodes_in_communities = len(set(node for nodes in communities.values() 
                                             for node in nodes))
        
        print(f"Total communities: {len(communities):,}")
        print(f"Community size statistics:")
        print(f"   Min: {min(community_sizes):,}")
        print(f"   Max: {max(community_sizes):,}")
        print(f"   Mean: {sum(community_sizes)/len(community_sizes):.2f}")
        print(f"   Median: {sorted(community_sizes)[len(community_sizes)//2]:,}")
        print(f"\nNode coverage:")
        print(f"   Unique nodes in communities: {unique_nodes_in_communities:,}")
        coverage_pct = (unique_nodes_in_communities / G_largest.number_of_nodes()) * 100
        print(f"   Coverage: {coverage_pct:.2f}% of graph nodes")
        
        # ====================================================================
        # FINAL SUMMARY
        # ====================================================================
        print_section("PIPELINE COMPLETED SUCCESSFULLY")
        
        print("✅ All steps completed!")
        print("\nGenerated files:")
        print("  📁 data/processed/")
        print("     - amazon_graph_cleaned.pkl")
        print("     - splits/train_graph.pkl")
        print("     - splits/positive_test_edges.pkl")
        print("     - splits/negative_test_edges.pkl")
        print("  📁 results/figures/")
        print("     - degree_distribution.png")
        print("  📁 results/tables/")
        print("     - network_statistics.csv")
        print("     - network_statistics.txt")
        
        print("\n" + "=" * 70)
        print("  READY FOR NEXT STEPS:")
        print("  - Module 2: Centrality Analysis")
        print("  - Community Detection")
        print("  - Link Prediction")
        print("=" * 70)
        
        return 0
    
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

