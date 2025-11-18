"""
Comprehensive visualization module for network analysis.

This module provides functions for visualizing networks, creating publication-quality
figures, and exporting data for external visualization tools.
"""

import os
import logging
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import pickle
import json

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


def plot_network_sample(
    G: nx.Graph,
    num_nodes: int = 1000,
    save_path: Optional[str] = None,
    layout: str = 'spring',
    color_by: str = 'degree',
    communities: Optional[Dict] = None,
    figsize: Tuple[int, int] = (12, 12),
    node_size: Union[int, List] = 50,
    edge_width: float = 0.5,
    alpha: float = 0.8
) -> None:
    """
    Plot a sample of the network with high-quality visualization.
    
    Args:
        G: NetworkX graph
        num_nodes: Number of nodes to sample (default: 1000)
        save_path: Optional path to save figure
        layout: Layout algorithm ('spring', 'circular', 'kamada_kawai', 'spectral')
        color_by: How to color nodes ('degree', 'community', 'random')
        communities: Optional dictionary mapping node to community ID
        figsize: Figure size tuple (default: (12, 12))
        node_size: Node size (int or list of sizes)
        edge_width: Edge width (default: 0.5)
        alpha: Transparency (default: 0.8)
    
    Raises:
        ValueError: If num_nodes is invalid or color_by is unknown
    """
    if num_nodes < 1:
        raise ValueError("num_nodes must be at least 1")
    
    if num_nodes > G.number_of_nodes():
        num_nodes = G.number_of_nodes()
        logger.warning(f"num_nodes reduced to {num_nodes} (graph size)")
    
    logger.info(f"Plotting network sample: {num_nodes} nodes, layout={layout}, color_by={color_by}")
    
    # Sample subgraph
    if num_nodes < G.number_of_nodes():
        nodes_sample = np.random.choice(list(G.nodes()), size=num_nodes, replace=False)
        G_sample = G.subgraph(nodes_sample).copy()
    else:
        G_sample = G
    
    logger.info(f"Sampled graph: {G_sample.number_of_nodes()} nodes, {G_sample.number_of_edges()} edges")
    
    # Compute layout
    logger.info(f"Computing {layout} layout...")
    if layout == 'spring':
        pos = nx.spring_layout(G_sample, k=1/np.sqrt(G_sample.number_of_nodes()), iterations=50, seed=42)
    elif layout == 'circular':
        pos = nx.circular_layout(G_sample)
    elif layout == 'kamada_kawai':
        try:
            pos = nx.kamada_kawai_layout(G_sample)
        except:
            logger.warning("Kamada-Kawai layout failed, using spring layout")
            pos = nx.spring_layout(G_sample, seed=42)
    elif layout == 'spectral':
        try:
            pos = nx.spectral_layout(G_sample)
        except:
            logger.warning("Spectral layout failed, using spring layout")
            pos = nx.spring_layout(G_sample, seed=42)
    else:
        pos = nx.spring_layout(G_sample, seed=42)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Determine node colors
    if color_by == 'degree':
        degrees = dict(G_sample.degree())
        node_colors = [degrees[node] for node in G_sample.nodes()]
        cmap = plt.cm.viridis
        colorbar_label = 'Degree'
    elif color_by == 'community':
        if communities is None:
            logger.warning("communities not provided, using degree coloring")
            degrees = dict(G_sample.degree())
            node_colors = [degrees[node] for node in G_sample.nodes()]
            cmap = plt.cm.viridis
            colorbar_label = 'Degree'
        else:
            # Map nodes to community IDs
            node_colors = [communities.get(node, -1) for node in G_sample.nodes()]
            cmap = plt.cm.tab20
            colorbar_label = 'Community'
    elif color_by == 'random':
        node_colors = np.random.rand(G_sample.number_of_nodes())
        cmap = plt.cm.viridis
        colorbar_label = 'Random'
    else:
        raise ValueError(f"Unknown color_by option: {color_by}")
    
    # Determine node sizes
    if isinstance(node_size, int):
        node_sizes = [node_size] * G_sample.number_of_nodes()
    else:
        node_sizes = node_size
    
    # Draw edges
    nx.draw_networkx_edges(
        G_sample, pos,
        alpha=alpha * 0.3,
        width=edge_width,
        ax=ax,
        edge_color='gray'
    )
    
    # Draw nodes
    nodes = nx.draw_networkx_nodes(
        G_sample, pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=alpha,
        cmap=cmap,
        ax=ax
    )
    
    # Add colorbar if using continuous colors
    if color_by in ['degree', 'random']:
        plt.colorbar(nodes, ax=ax, label=colorbar_label)
    
    # Remove axes
    ax.axis('off')
    ax.set_title(f'Network Sample ({G_sample.number_of_nodes()} nodes, {G_sample.number_of_edges()} edges)',
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Network sample plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_ego_network(
    G: nx.Graph,
    center_node: Union[int, str],
    radius: int = 2,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 10),
    highlight_center: bool = True
) -> None:
    """
    Plot ego network (neighborhood) around a center node.
    
    Args:
        G: NetworkX graph
        center_node: Center node ID
        radius: Radius of ego network (default: 2)
        save_path: Optional path to save figure
        figsize: Figure size tuple (default: (10, 10))
        highlight_center: Whether to highlight center node (default: True)
    
    Raises:
        ValueError: If center_node not in graph
    """
    if center_node not in G:
        raise ValueError(f"Center node {center_node} not in graph")
    
    logger.info(f"Plotting ego network: center={center_node}, radius={radius}")
    
    # Extract ego network
    if radius == 1:
        ego_nodes = list(G.neighbors(center_node)) + [center_node]
    else:
        # Use BFS to get nodes within radius
        ego_nodes = [center_node]
        current_level = {center_node}
        for _ in range(radius):
            next_level = set()
            for node in current_level:
                next_level.update(G.neighbors(node))
            ego_nodes.extend(next_level - set(ego_nodes))
            current_level = next_level
    
    G_ego = G.subgraph(ego_nodes).copy()
    
    logger.info(f"Ego network: {G_ego.number_of_nodes()} nodes, {G_ego.number_of_edges()} edges")
    
    # Compute layout
    pos = nx.spring_layout(G_ego, k=1, iterations=50, seed=42)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw edges
    nx.draw_networkx_edges(G_ego, pos, alpha=0.3, width=0.5, ax=ax, edge_color='gray')
    
    # Draw non-center nodes
    non_center_nodes = [n for n in G_ego.nodes() if n != center_node]
    if non_center_nodes:
        nx.draw_networkx_nodes(
            G_ego, pos,
            nodelist=non_center_nodes,
            node_color='lightblue',
            node_size=300,
            alpha=0.7,
            ax=ax
        )
    
    # Draw center node
    if highlight_center:
        nx.draw_networkx_nodes(
            G_ego, pos,
            nodelist=[center_node],
            node_color='red',
            node_size=800,
            alpha=1.0,
            ax=ax
        )
    
    # Draw labels
    labels = {center_node: str(center_node)}
    nx.draw_networkx_labels(G_ego, pos, labels, font_size=10, font_weight='bold', ax=ax)
    
    # Remove axes
    ax.axis('off')
    ax.set_title(f'Ego Network: Node {center_node} (Radius {radius})',
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Ego network plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_all_visualizations(
    results_dir: str,
    output_dir: str
) -> None:
    """
    Load all results and generate comprehensive visualizations.
    
    Args:
        results_dir: Directory containing result files
        output_dir: Directory to save visualizations
    
    Raises:
        ValueError: If directories don't exist
    """
    results_path = Path(results_dir)
    output_path = Path(output_dir)
    
    if not results_path.exists():
        raise ValueError(f"Results directory does not exist: {results_dir}")
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating all visualizations from {results_dir} to {output_dir}")
    
    # Load graph
    graph_path = results_path / 'amazon_graph_cleaned.pkl'
    if graph_path.exists():
        logger.info("Loading graph...")
        with open(graph_path, 'rb') as f:
            G = pickle.load(f)
        
        # Network sample visualization
        logger.info("Creating network sample visualization...")
        plot_network_sample(
            G,
            num_nodes=1000,
            save_path=str(output_path / 'network_sample.png'),
            layout='spring',
            color_by='degree'
        )
        
        # Ego network visualization (sample a high-degree node)
        if G.number_of_nodes() > 0:
            degrees = dict(G.degree())
            top_node = max(degrees.items(), key=lambda x: x[1])[0]
            logger.info(f"Creating ego network for node {top_node}...")
            plot_ego_network(
                G,
                top_node,
                radius=2,
                save_path=str(output_path / f'ego_network_node_{top_node}.png')
            )
    
    # Load centrality results
    centrality_dir = results_path / 'centrality'
    if centrality_dir.exists():
        logger.info("Loading centrality results...")
        # Could load and visualize centrality distributions here
    
    # Load community results
    community_dir = results_path / 'communities'
    if community_dir.exists():
        logger.info("Loading community results...")
        # Could load and visualize communities here
    
    # Load link prediction results
    link_pred_dir = results_path / 'link_prediction'
    if link_pred_dir.exists():
        logger.info("Loading link prediction results...")
        # Could load and visualize link prediction results here
    
    logger.info(f"All visualizations created in {output_dir}")


def export_to_gephi(
    G: nx.Graph,
    communities: Optional[Dict] = None,
    output_path: str = 'network.gexf',
    node_attributes: Optional[Dict] = None,
    edge_attributes: Optional[Dict] = None
) -> None:
    """
    Export graph to GEXF format for Gephi visualization.
    
    Args:
        G: NetworkX graph
        communities: Optional dictionary mapping node to community ID
        output_path: Path to save GEXF file
        node_attributes: Optional dictionary of node attributes
        edge_attributes: Optional dictionary of edge attributes
    
    Raises:
        ValueError: If output_path is invalid
    """
    if not output_path.endswith('.gexf'):
        raise ValueError("output_path must end with .gexf")
    
    logger.info(f"Exporting graph to Gephi format: {output_path}")
    
    # Create a copy for export
    G_export = G.copy()
    
    # Add community assignments as node attributes
    if communities:
        logger.info("Adding community assignments...")
        nx.set_node_attributes(G_export, communities, 'community')
    
    # Add other node attributes
    if node_attributes:
        for attr_name, attr_dict in node_attributes.items():
            nx.set_node_attributes(G_export, attr_dict, attr_name)
    
    # Add edge attributes
    if edge_attributes:
        for attr_name, attr_dict in edge_attributes.items():
            nx.set_edge_attributes(G_export, attr_dict, attr_name)
    
    # Add degree as attribute
    degrees = dict(G_export.degree())
    nx.set_node_attributes(G_export, degrees, 'degree')
    
    # Write GEXF file
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    nx.write_gexf(G_export, output_path)
    
    logger.info(f"Graph exported: {G_export.number_of_nodes()} nodes, {G_export.number_of_edges()} edges")
    logger.info(f"GEXF file saved to {output_path}")


def plot_degree_heatmap(
    G: nx.Graph,
    communities: Optional[Dict] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Create heatmap showing degree distribution by community.
    
    Args:
        G: NetworkX graph
        communities: Optional dictionary mapping node to community ID
        save_path: Optional path to save figure
        figsize: Figure size tuple (default: (12, 8))
    """
    logger.info("Creating degree heatmap...")
    
    if communities is None:
        logger.warning("No communities provided, skipping heatmap")
        return
    
    # Create DataFrame with node, degree, community
    data = []
    degrees = dict(G.degree())
    for node in G.nodes():
        data.append({
            'node': node,
            'degree': degrees.get(node, 0),
            'community': communities.get(node, -1)
        })
    
    df = pd.DataFrame(data)
    
    # Create pivot table: community vs degree bins
    df['degree_bin'] = pd.cut(df['degree'], bins=20, labels=False)
    heatmap_data = df.pivot_table(
        values='node',
        index='community',
        columns='degree_bin',
        aggfunc='count',
        fill_value=0
    )
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(heatmap_data, cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Number of Nodes'})
    ax.set_xlabel('Degree Bin', fontsize=12)
    ax.set_ylabel('Community', fontsize=12)
    ax.set_title('Degree Distribution by Community', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Degree heatmap saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """
    Main function demonstrating usage of the visualization module.
    """
    print("=" * 60)
    print("Visualization Module - Demo")
    print("=" * 60)
    
    try:
        import networkx as nx
        
        # Create a sample graph
        print("\n1. Creating sample graph...")
        G = nx.erdos_renyi_graph(n=500, p=0.02, seed=42)
        print(f"   Created graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Test network sample plot
        print("\n2. Creating network sample plot...")
        plot_network_sample(
            G,
            num_nodes=200,
            save_path='results/figures/network_sample_demo.png',
            layout='spring',
            color_by='degree'
        )
        print("   ✅ Network sample plot created")
        
        # Test ego network plot
        print("\n3. Creating ego network plot...")
        if G.number_of_nodes() > 0:
            center_node = list(G.nodes())[0]
            plot_ego_network(
                G,
                center_node,
                radius=2,
                save_path='results/figures/ego_network_demo.png'
            )
            print(f"   ✅ Ego network plot created for node {center_node}")
        
        # Test Gephi export
        print("\n4. Exporting to Gephi format...")
        communities = {node: node % 5 for node in G.nodes()}  # Dummy communities
        export_to_gephi(
            G,
            communities=communities,
            output_path='results/network_demo.gexf'
        )
        print("   ✅ GEXF file created")
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)
    
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"\n✗ Error: {e}")
        raise


if __name__ == "__main__":
    main()

