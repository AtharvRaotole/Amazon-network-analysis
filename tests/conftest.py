"""
Pytest configuration and shared fixtures.

This file contains shared fixtures and configuration for all tests.
"""

import pytest
import networkx as nx
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


@pytest.fixture(scope="session")
def test_data_dir():
    """Create test data directory."""
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


@pytest.fixture(scope="session")
def sample_graph_large():
    """Create a larger sample graph for performance testing."""
    return nx.erdos_renyi_graph(n=500, p=0.02, seed=42)


@pytest.fixture(scope="session")
def sample_graph_small():
    """Create a small sample graph for quick tests."""
    return nx.erdos_renyi_graph(n=20, p=0.3, seed=42)


@pytest.fixture(scope="session")
def sample_communities_dict():
    """Create sample communities as dictionary."""
    return {
        0: 1, 1: 1, 2: 1,
        3: 2, 4: 2, 5: 2,
        6: 3, 7: 3, 8: 3, 9: 3
    }

