"""
Parallel processing module for network analysis.

This module provides functions for parallelizing network analysis algorithms
and benchmarking their performance.
"""

import os
import time
import logging
import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Callable
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

# Try to import memory_profiler (optional)
try:
    from memory_profiler import profile
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False
    logging.warning("memory_profiler library not available. Memory profiling will be limited.")

# Import link prediction functions
from link_prediction import (
    common_neighbors_score,
    adamic_adar_score,
    jaccard_coefficient_score
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _get_n_jobs(n_jobs: int) -> int:
    """
    Get number of jobs for parallel processing.
    
    Args:
        n_jobs: Number of jobs (-1 for all cores, positive for specific number)
    
    Returns:
        Number of jobs to use
    """
    if n_jobs == -1:
        return multiprocessing.cpu_count()
    elif n_jobs < 0:
        return max(1, multiprocessing.cpu_count() + n_jobs + 1)
    else:
        return max(1, n_jobs)


def parallel_pagerank(
    G: nx.Graph,
    n_jobs: int = -1,
    alpha: float = 0.85,
    max_iter: int = 100
) -> Dict:
    """
    Compute PageRank with parallelization if possible.
    
    Note: NetworkX PageRank is already optimized, but we can compare
    serial vs parallel implementations for custom algorithms.
    
    Args:
        G: NetworkX graph
        n_jobs: Number of parallel jobs (-1 for all cores)
        alpha: Damping parameter for PageRank
        max_iter: Maximum iterations
    
    Returns:
        Dictionary with:
            - pagerank: PageRank scores
            - serial_time: Time for serial computation
            - parallel_time: Time for parallel computation (if applicable)
            - speedup: Speedup factor
    """
    logger.info(f"Computing PageRank (n_jobs={n_jobs}, alpha={alpha})...")
    
    # Serial PageRank (NetworkX implementation)
    start_time = time.time()
    pagerank_serial = nx.pagerank(G, alpha=alpha, max_iter=max_iter)
    serial_time = time.time() - start_time
    
    # Note: NetworkX PageRank is already optimized
    # For demonstration, we'll use the serial version
    # In practice, parallel PageRank requires custom implementation
    parallel_time = serial_time  # No parallel version available in NetworkX
    speedup = 1.0
    
    logger.info(f"PageRank computed: {serial_time:.4f} seconds")
    
    return {
        'pagerank': pagerank_serial,
        'serial_time': serial_time,
        'parallel_time': parallel_time,
        'speedup': speedup,
        'num_nodes': G.number_of_nodes()
    }


def _compute_betweenness_batch(
    G: nx.Graph,
    nodes: List,
    k: Optional[int] = None
) -> Dict:
    """
    Compute betweenness centrality for a batch of nodes.
    
    Args:
        G: NetworkX graph
        nodes: List of nodes to compute betweenness for
        k: Number of nodes to sample (None for all)
    
    Returns:
        Dictionary of {node: betweenness_score}
    """
    # Compute betweenness for the whole graph (or sample k nodes)
    # Then filter to only return values for nodes in this batch
    if k is not None and k < G.number_of_nodes():
        # Sample k nodes from entire graph for betweenness computation
        all_nodes = list(G.nodes())
        sampled_nodes = np.random.choice(all_nodes, size=k, replace=False)
        betweenness = nx.betweenness_centrality(G, k=k)
    else:
        betweenness = nx.betweenness_centrality(G, k=k)
    
    # Return only the nodes in this batch
    return {node: betweenness.get(node, 0.0) for node in nodes if node in betweenness}


def parallel_betweenness(
    G: nx.Graph,
    n_jobs: int = -1,
    k: int = 1000
) -> Dict:
    """
    Compute betweenness centrality in parallel.
    
    Args:
        G: NetworkX graph
        n_jobs: Number of parallel jobs (-1 for all cores)
        k: Number of nodes to sample for betweenness
    
    Returns:
        Dictionary with:
            - betweenness: Betweenness centrality scores
            - serial_time: Time for serial computation
            - parallel_time: Time for parallel computation
            - speedup: Speedup factor
    """
    logger.info(f"Computing betweenness centrality (n_jobs={n_jobs}, k={k})...")
    
    nodes = list(G.nodes())
    num_nodes = len(nodes)
    
    # Serial version
    start_time = time.time()
    betweenness_serial = nx.betweenness_centrality(G, k=k)
    serial_time = time.time() - start_time
    
    # Parallel version: split nodes into batches
    n_jobs_actual = _get_n_jobs(n_jobs)
    
    if num_nodes > n_jobs_actual and n_jobs_actual > 1:
        # Split nodes into batches
        batch_size = max(1, num_nodes // n_jobs_actual)
        batches = [nodes[i:i+batch_size] for i in range(0, num_nodes, batch_size)]
        
        start_time = time.time()
        results = Parallel(n_jobs=n_jobs_actual, verbose=0)(
            delayed(_compute_betweenness_batch)(G, batch, k=None)
            for batch in batches
        )
        
        # Combine results
        betweenness_parallel = {}
        for result in results:
            betweenness_parallel.update(result)
        
        parallel_time = time.time() - start_time
        speedup = serial_time / parallel_time if parallel_time > 0 else 1.0
    else:
        betweenness_parallel = betweenness_serial
        parallel_time = serial_time
        speedup = 1.0
    
    logger.info(f"Betweenness computed: serial={serial_time:.4f}s, parallel={parallel_time:.4f}s, speedup={speedup:.2f}x")
    
    return {
        'betweenness': betweenness_parallel,
        'serial_time': serial_time,
        'parallel_time': parallel_time,
        'speedup': speedup,
        'num_nodes': num_nodes,
        'k': k
    }


def _compute_similarity_batch(
    G: nx.Graph,
    edge_batch: List[Tuple],
    method: str
) -> List[float]:
    """
    Compute similarity scores for a batch of edges.
    
    Args:
        G: NetworkX graph
        edge_batch: List of (u, v) tuples
        method: Similarity method ('common_neighbors', 'adamic_adar', 'jaccard')
    
    Returns:
        List of scores
    """
    if method == 'common_neighbors':
        return common_neighbors_score(G, edge_batch)
    elif method == 'adamic_adar':
        return adamic_adar_score(G, edge_batch)
    elif method == 'jaccard':
        return jaccard_coefficient_score(G, edge_batch)
    else:
        raise ValueError(f"Unknown method: {method}")


def parallel_link_prediction(
    G: nx.Graph,
    edge_list: List[Tuple[int, int]],
    method: str = 'adamic_adar',
    n_jobs: int = -1,
    batch_size: Optional[int] = None
) -> Dict:
    """
    Compute link prediction scores in parallel.
    
    Args:
        G: NetworkX graph
        edge_list: List of (u, v) tuples to score
        method: Similarity method ('common_neighbors', 'adamic_adar', 'jaccard')
        n_jobs: Number of parallel jobs (-1 for all cores)
        batch_size: Size of batches (None for automatic)
    
    Returns:
        Dictionary with:
            - scores: List of scores
            - serial_time: Time for serial computation
            - parallel_time: Time for parallel computation
            - speedup: Speedup factor
    """
    logger.info(f"Computing link prediction scores (method={method}, n_jobs={n_jobs}, edges={len(edge_list)})...")
    
    # Serial version
    start_time = time.time()
    scores_serial = _compute_similarity_batch(G, edge_list, method)
    serial_time = time.time() - start_time
    
    # Parallel version
    n_jobs_actual = _get_n_jobs(n_jobs)
    num_edges = len(edge_list)
    
    if num_edges > n_jobs_actual and n_jobs_actual > 1:
        # Determine batch size
        if batch_size is None:
            batch_size = max(1, num_edges // (n_jobs_actual * 4))  # 4 batches per job
        
        # Split edges into batches
        batches = [edge_list[i:i+batch_size] for i in range(0, num_edges, batch_size)]
        
        start_time = time.time()
        results = Parallel(n_jobs=n_jobs_actual, verbose=0)(
            delayed(_compute_similarity_batch)(G, batch, method)
            for batch in tqdm(batches, desc="Processing batches")
        )
        
        # Flatten results
        scores_parallel = [score for batch_scores in results for score in batch_scores]
        parallel_time = time.time() - start_time
        speedup = serial_time / parallel_time if parallel_time > 0 else 1.0
    else:
        scores_parallel = scores_serial
        parallel_time = serial_time
        speedup = 1.0
    
    logger.info(f"Link prediction computed: serial={serial_time:.4f}s, parallel={parallel_time:.4f}s, speedup={speedup:.2f}x")
    
    return {
        'scores': scores_parallel,
        'serial_time': serial_time,
        'parallel_time': parallel_time,
        'speedup': speedup,
        'num_edges': num_edges,
        'method': method
    }


def benchmark_scalability(
    G: nx.Graph,
    algorithms: List[str],
    sample_sizes: List[float] = [0.1, 0.25, 0.5, 0.75, 1.0],
    n_jobs: int = -1
) -> pd.DataFrame:
    """
    Benchmark algorithms on different graph sizes.
    
    Args:
        G: NetworkX graph
        algorithms: List of algorithm names to test ('pagerank', 'betweenness', 'link_prediction')
        sample_sizes: List of sample ratios (0.0 to 1.0)
        n_jobs: Number of parallel jobs
    
    Returns:
        DataFrame with columns: algorithm, sample_size, num_nodes, num_edges, time, memory_peak
    """
    logger.info(f"Benchmarking scalability for {len(algorithms)} algorithms...")
    
    results = []
    nodes = list(G.nodes())
    num_nodes_full = len(nodes)
    
    for sample_ratio in sample_sizes:
        # Sample graph
        num_nodes_sample = int(num_nodes_full * sample_ratio)
        if num_nodes_sample < 1:
            num_nodes_sample = 1
        
        sampled_nodes = np.random.choice(nodes, size=num_nodes_sample, replace=False)
        G_sample = G.subgraph(sampled_nodes).copy()
        
        num_edges_sample = G_sample.number_of_edges()
        
        logger.info(f"Testing on {num_nodes_sample} nodes ({sample_ratio*100:.1f}% of original)...")
        
        for algo in algorithms:
            try:
                start_time = time.time()
                
                if algo == 'pagerank':
                    result = parallel_pagerank(G_sample, n_jobs=n_jobs)
                    exec_time = result['serial_time']
                    memory_peak = 0  # Would need memory_profiler
                
                elif algo == 'betweenness':
                    k = min(1000, num_nodes_sample)
                    result = parallel_betweenness(G_sample, n_jobs=n_jobs, k=k)
                    exec_time = result['parallel_time']
                    memory_peak = 0
                
                elif algo == 'link_prediction':
                    # Create sample edge list
                    edge_list = list(G_sample.edges())[:min(1000, G_sample.number_of_edges())]
                    result = parallel_link_prediction(G_sample, edge_list, method='adamic_adar', n_jobs=n_jobs)
                    exec_time = result['parallel_time']
                    memory_peak = 0
                
                else:
                    logger.warning(f"Unknown algorithm: {algo}")
                    continue
                
                results.append({
                    'algorithm': algo,
                    'sample_size': sample_ratio,
                    'num_nodes': num_nodes_sample,
                    'num_edges': num_edges_sample,
                    'time': exec_time,
                    'memory_peak': memory_peak
                })
                
                logger.info(f"  {algo}: {exec_time:.4f} seconds")
            
            except Exception as e:
                logger.error(f"Error benchmarking {algo} at {sample_ratio}: {e}")
                continue
    
    df = pd.DataFrame(results)
    logger.info(f"Benchmarking complete: {len(df)} measurements")
    
    return df


def profile_memory_usage(
    func: Callable,
    *args,
    **kwargs
) -> Dict:
    """
    Profile memory usage of a function.
    
    Args:
        func: Function to profile
        *args: Positional arguments for function
        **kwargs: Keyword arguments for function
    
    Returns:
        Dictionary with:
            - result: Function result
            - memory_peak: Peak memory usage (MB)
            - memory_increment: Memory increment (MB)
            - execution_time: Execution time (seconds)
    """
    logger.info(f"Profiling memory usage for {func.__name__}...")
    
    if MEMORY_PROFILER_AVAILABLE:
        try:
            import tracemalloc
            
            # Start memory tracking
            tracemalloc.start()
            start_memory = tracemalloc.get_traced_memory()[0] / 1024 / 1024  # MB
            
            # Execute function
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Get peak memory
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            memory_peak = peak / 1024 / 1024  # MB
            memory_increment = (peak - start_memory) / 1024 / 1024  # MB
            
            logger.info(f"Memory profiling: peak={memory_peak:.2f} MB, increment={memory_increment:.2f} MB")
        
        except Exception as e:
            logger.warning(f"Memory profiling failed: {e}")
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            memory_peak = 0.0
            memory_increment = 0.0
    
    else:
        # Fallback: just measure time
        logger.warning("memory_profiler not available, only timing will be measured")
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        memory_peak = 0.0
        memory_increment = 0.0
    
    return {
        'result': result,
        'memory_peak': memory_peak,
        'memory_increment': memory_increment,
        'execution_time': execution_time
    }


def main():
    """
    Main function demonstrating usage of the parallel processing module.
    """
    print("=" * 60)
    print("Parallel Processing Module - Demo")
    print("=" * 60)
    
    try:
        import networkx as nx
        
        # Create a sample graph
        print("\n1. Creating sample graph...")
        G = nx.erdos_renyi_graph(n=1000, p=0.01, seed=42)
        print(f"   Created graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Test parallel PageRank
        print("\n2. Testing parallel PageRank...")
        pr_result = parallel_pagerank(G, n_jobs=-1)
        print(f"   Serial time: {pr_result['serial_time']:.4f} seconds")
        print(f"   Speedup: {pr_result['speedup']:.2f}x")
        
        # Test parallel betweenness
        print("\n3. Testing parallel betweenness...")
        bt_result = parallel_betweenness(G, n_jobs=-1, k=100)
        print(f"   Serial time: {bt_result['serial_time']:.4f} seconds")
        print(f"   Parallel time: {bt_result['parallel_time']:.4f} seconds")
        print(f"   Speedup: {bt_result['speedup']:.2f}x")
        
        # Test parallel link prediction
        print("\n4. Testing parallel link prediction...")
        edge_list = list(G.edges())[:500]
        lp_result = parallel_link_prediction(G, edge_list, method='adamic_adar', n_jobs=-1)
        print(f"   Serial time: {lp_result['serial_time']:.4f} seconds")
        print(f"   Parallel time: {lp_result['parallel_time']:.4f} seconds")
        print(f"   Speedup: {lp_result['speedup']:.2f}x")
        
        # Test scalability benchmark
        print("\n5. Testing scalability benchmark...")
        benchmark_df = benchmark_scalability(
            G,
            algorithms=['pagerank', 'betweenness'],
            sample_sizes=[0.1, 0.5, 1.0],
            n_jobs=-1
        )
        print(benchmark_df.to_string(index=False))
        
        # Test memory profiling
        print("\n6. Testing memory profiling...")
        def test_func():
            return nx.pagerank(G)
        
        mem_result = profile_memory_usage(test_func)
        print(f"   Execution time: {mem_result['execution_time']:.4f} seconds")
        if mem_result['memory_peak'] > 0:
            print(f"   Peak memory: {mem_result['memory_peak']:.2f} MB")
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)
    
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"\n✗ Error: {e}")
        raise


if __name__ == "__main__":
    main()

