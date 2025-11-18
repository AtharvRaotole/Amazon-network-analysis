# Large-Scale Link Analysis: Amazon Product Co-Purchasing Network

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NetworkX](https://img.shields.io/badge/networkx-3.2+-green.svg)](https://networkx.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A comprehensive network analysis project implementing large-scale link analysis algorithms on the Amazon Product Co-Purchasing Network. This project covers centrality analysis, community detection, link prediction, and advanced spam detection techniques based on textbook algorithms from "Mining of Massive Datasets".

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Modules](#modules)
- [Notebooks](#notebooks)
- [Results](#results)
- [Performance](#performance)
- [Contributing](#contributing)
- [References](#references)

## 🎯 Overview

This project implements a complete pipeline for analyzing large-scale networks, specifically the Amazon Product Co-Purchasing Network from Stanford's SNAP dataset. The analysis includes:

- **Centrality Measures**: PageRank, Degree Centrality, Betweenness Centrality, HITS
- **Community Detection**: Louvain, Label Propagation, Greedy Modularity
- **Link Prediction**: Similarity-based and Machine Learning approaches
- **Spam Detection**: TrustRank, Spam Mass, Structural Pattern Detection
- **Spam Farm Analysis**: Creation, effectiveness analysis, and detection evaluation

The project is designed for both research and educational purposes, with comprehensive documentation, Jupyter notebooks, and production-ready code.

## ✨ Features

### Core Analysis
- ✅ **Data Loading & Preprocessing**: Efficient graph loading, cleaning, and train/test splitting
- ✅ **Centrality Analysis**: Multiple centrality measures with sampling for large graphs
- ✅ **Community Detection**: Three state-of-the-art algorithms with evaluation metrics
- ✅ **Link Prediction**: Both similarity-based and ML-based approaches with comprehensive evaluation

### Advanced Spam Analysis
- ✅ **Spam Farm Generation**: Multiple spam farm types (simple, multiple, collaborative)
- ✅ **TrustRank Implementation**: Topic-sensitive PageRank for spam detection
- ✅ **Spam Mass Calculation**: Quantitative measure of spam contribution to PageRank
- ✅ **Structural Detection**: Pattern-based detection (star, honeypot, reciprocal links)
- ✅ **Effectiveness Analysis**: ROI, amplification, and network damage assessment

### Infrastructure
- ✅ **Comprehensive Testing**: Full test suite with 11+ test modules
- ✅ **HPC Support**: SLURM job scripts for high-performance computing
- ✅ **Parallel Processing**: Joblib-based parallelization for scalability
- ✅ **Visualization**: Publication-quality plots and interactive visualizations
- ✅ **Reporting**: Automated report generation (HTML, Markdown, LaTeX)

## 📊 Dataset

### Amazon Product Co-Purchasing Network

- **Source**: [Stanford SNAP](https://snap.stanford.edu/data/bigdata/communities/com-amazon.ungraph.txt.gz)
- **Nodes**: 334,863 products
- **Edges**: 925,872 co-purchasing relationships
- **Type**: Undirected graph
- **Ground Truth**: 75,149 communities

### Data Files

The dataset includes:
- `com-amazon.ungraph.txt.gz`: Main graph file (edge list)
- `com-amazon.all.cmty.txt.gz`: Ground-truth communities

**Note**: The full dataset is large (~50MB compressed). For testing, the system can work with sample graphs, but for complete analysis, download the full dataset.

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB+ RAM (8GB+ recommended for full dataset)
- Git (for cloning the repository)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/AtharvRaotole/Amazon-network-analysis.git
   cd Amazon-network-analysis
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download dataset** (optional, for full analysis)
   ```bash
   bash setup.sh  # Automatically downloads and extracts dataset
   ```

5. **Verify installation**
   ```bash
   python tests/verify_setup.py
   ```

### Manual Dataset Download

If you prefer to download manually:

```bash
mkdir -p data/raw
cd data/raw
wget https://snap.stanford.edu/data/bigdata/communities/com-amazon.ungraph.txt.gz
wget https://snap.stanford.edu/data/bigdata/communities/com-amazon.all.cmty.txt.gz
gunzip *.gz
```

## 📁 Project Structure

```
finalbda/
├── data/
│   ├── raw/              # Raw dataset files
│   ├── processed/        # Preprocessed graphs (pickle)
│   └── results/          # Analysis results
│       ├── centrality/   # Centrality analysis results
│       ├── communities/  # Community detection results
│       ├── link_prediction/  # Link prediction results
│       ├── spam_analysis/    # Spam analysis results
│       ├── figures/     # Generated plots
│       └── tables/      # Data tables
│
├── src/                  # Source code modules
│   ├── data_loader.py           # Data loading utilities
│   ├── preprocessing.py         # Graph preprocessing
│   ├── exploratory_analysis.py  # Initial data exploration
│   ├── centrality_analysis.py   # Centrality measures
│   ├── centrality_visualization.py  # Centrality plots
│   ├── community_detection.py   # Community algorithms
│   ├── community_evaluation.py  # Community evaluation
│   ├── community_visualization.py  # Community plots
│   ├── link_prediction.py      # Similarity-based link prediction
│   ├── ml_link_prediction.py   # ML-based link prediction
│   ├── link_prediction_evaluation.py  # Link prediction evaluation
│   ├── network_properties.py   # Network property analysis
│   ├── parallel_processing.py  # Parallel computation
│   ├── performance_analysis.py # Performance benchmarking
│   ├── visualization.py        # General visualizations
│   ├── report_generator.py     # Report generation
│   ├── spam_farm_generator.py  # Spam farm creation
│   ├── spam_farm_variants.py  # Spam farm variations
│   ├── trustrank.py            # TrustRank implementation
│   ├── spam_mass.py            # Spam mass calculation
│   ├── structural_spam_detection.py  # Pattern-based detection
│   ├── spam_detection_evaluation.py   # Detection evaluation
│   ├── spam_effectiveness_analysis.py  # Effectiveness analysis
│   └── spam_analysis_pipeline.py     # Complete spam pipeline
│
├── notebooks/            # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_centrality_analysis.ipynb
│   ├── 03_community_detection.ipynb
│   ├── 04_link_prediction.ipynb
│   └── 05_spam_farm_analysis.ipynb
│
├── tests/                # Test suite
│   ├── verify_setup.py   # Setup verification
│   ├── test_modules.py   # Unit tests
│   └── conftest.py       # Test fixtures
│
├── slurm_jobs/           # HPC job scripts
│   ├── run_centrality.slurm
│   ├── run_communities.slurm
│   ├── run_link_prediction.slurm
│   ├── run_full_pipeline.slurm
│   └── submit_all.sh
│
├── requirements.txt      # Python dependencies
├── setup.sh             # Automated setup script
├── main.py              # Main execution script
├── run_full_system_test.py  # Comprehensive system test
└── README.md            # This file
```

## 💻 Usage

### Basic Usage

#### 1. Data Exploration
```python
from src.data_loader import load_graph, load_communities
from src.preprocessing import get_largest_component

# Load dataset
G = load_graph('data/raw/com-amazon.ungraph.txt')
communities = load_communities('data/raw/com-amazon.all.cmty.txt')

# Preprocess
G_clean = get_largest_component(G)
print(f"Graph: {len(G_clean)} nodes, {G_clean.number_of_edges()} edges")
```

#### 2. Centrality Analysis
```python
from src.centrality_analysis import (
    compute_pagerank, compute_degree_centrality,
    compute_betweenness_centrality, compute_hits
)

pr = compute_pagerank(G_clean)
dc = compute_degree_centrality(G_clean)
bc = compute_betweenness_centrality(G_clean, k=1000)  # Sampling for large graphs
hubs, auth = compute_hits(G_clean)
```

#### 3. Community Detection
```python
from src.community_detection import detect_communities_louvain
from src.community_evaluation import evaluate_communities

communities = detect_communities_louvain(G_clean)
evaluation = evaluate_communities(communities, ground_truth, G_clean)
print(f"NMI: {evaluation['nmi']:.4f}, ARI: {evaluation['ari']:.4f}")
```

#### 4. Link Prediction
```python
from src.link_prediction import adamic_adar_score
from src.ml_link_prediction import train_random_forest

# Similarity-based
scores = adamic_adar_score(G_train, test_edges)

# ML-based
model = train_random_forest(X_train, y_train)
predictions = model.predict_proba(X_test)
```

#### 5. Spam Analysis
```python
from src.spam_analysis_pipeline import run_complete_spam_experiment

config = {
    'spam_type': 'simple',
    'm': 1000,
    'external_links': 10,
    'beta': 0.85,
    'detection_methods': ['trustrank', 'spam_mass', 'structural']
}

results = run_complete_spam_experiment(G_original, config)
```

### Command Line Interface

#### Run Full Pipeline
```bash
python main.py --data-dir ./data --output-dir ./results --modules all
```

#### Run Specific Module
```bash
python main.py --modules centrality,communities --parallel --n-jobs 4
```

#### Run System Tests
```bash
python run_full_system_test.py
```

### Jupyter Notebooks

Launch Jupyter and explore the notebooks:

```bash
jupyter notebook notebooks/
```

Each notebook provides:
- Step-by-step analysis
- Visualizations
- Interpretations
- Interactive exploration

## 🔧 Modules

### Core Modules

| Module | Description | Key Functions |
|--------|-------------|---------------|
| `data_loader.py` | Dataset loading and saving | `load_graph()`, `load_communities()`, `save_graph()` |
| `preprocessing.py` | Graph cleaning and splitting | `remove_self_loops()`, `get_largest_component()`, `create_train_test_split()` |
| `exploratory_analysis.py` | Initial data exploration | `degree_distribution()`, `compute_network_stats()` |
| `centrality_analysis.py` | Centrality measures | `compute_pagerank()`, `compute_betweenness_centrality()`, `compute_hits()` |
| `community_detection.py` | Community algorithms | `detect_communities_louvain()`, `detect_communities_label_propagation()` |
| `link_prediction.py` | Similarity-based prediction | `adamic_adar_score()`, `jaccard_coefficient_score()` |
| `ml_link_prediction.py` | ML-based prediction | `train_random_forest()`, `extract_edge_features()` |

### Spam Analysis Modules

| Module | Description | Key Functions |
|--------|-------------|---------------|
| `spam_farm_generator.py` | Spam farm creation | `create_simple_spam_farm()`, `create_multiple_spam_farms()` |
| `trustrank.py` | TrustRank computation | `compute_trustrank()`, `select_trusted_pages()` |
| `spam_mass.py` | Spam mass calculation | `calculate_spam_mass()`, `classify_by_spam_mass()` |
| `structural_spam_detection.py` | Pattern detection | `detect_star_patterns()`, `detect_honeypot_patterns()` |
| `spam_effectiveness_analysis.py` | Effectiveness analysis | `calculate_pagerank_amplification()`, `analyze_spam_farm_roi()` |
| `spam_analysis_pipeline.py` | Complete pipeline | `run_complete_spam_experiment()`, `run_parameter_sweep()` |

## 📓 Notebooks

1. **01_data_exploration.ipynb**
   - Dataset loading and preprocessing
   - Basic statistics and visualizations
   - Degree distribution analysis

2. **02_centrality_analysis.ipynb**
   - All centrality measures computation
   - Top-k node analysis
   - Correlation and overlap analysis

3. **03_community_detection.ipynb**
   - Community detection algorithms
   - Evaluation against ground truth
   - Community visualization

4. **04_link_prediction.ipynb**
   - Similarity-based methods
   - ML-based approach
   - Comprehensive evaluation

5. **05_spam_farm_analysis.ipynb**
   - Spam farm creation and impact
   - TrustRank and Spam Mass detection
   - Method comparison and evaluation

## 📈 Results

Results are automatically saved to the `results/` directory:

- **Centrality Results**: `results/centrality/*.csv`
- **Community Results**: `results/communities/*.csv`
- **Link Prediction**: `results/link_prediction/*.csv`
- **Spam Analysis**: `results/spam_analysis/*.json`
- **Visualizations**: `results/figures/*.png`
- **Summary**: `results/summary.json`

### Example Results

- **PageRank Amplification**: 31.75x boost observed for spam farms
- **Detection Performance**: Precision=1.0, Recall=1.0, F1=1.0 (on test spam farms)
- **Community Detection**: Louvain achieves highest modularity (0.28+)
- **Link Prediction**: Random Forest achieves AUC-ROC > 0.85

## ⚡ Performance

### Scalability

- **Sampling**: Betweenness centrality uses sampling (k=1000) for large graphs
- **Parallel Processing**: Joblib-based parallelization for computationally intensive tasks
- **Memory Efficiency**: Batch processing for link prediction on large edge sets

### Benchmarks

On a standard laptop (8GB RAM, 4 cores):
- **PageRank**: ~0.1s for 1K nodes, ~10s for 100K nodes
- **Community Detection**: ~0.1s for 1K nodes, ~60s for 100K nodes
- **Link Prediction**: ~1s per 1K edges

For the full Amazon dataset (334K nodes), use HPC resources with provided SLURM scripts.

## 🧪 Testing

### Run All Tests
```bash
python run_full_system_test.py
```

### Run Unit Tests
```bash
pytest tests/test_modules.py -v
```

### Verify Setup
```bash
python tests/verify_setup.py
```

## 🖥️ HPC Usage

For large-scale analysis, use the provided SLURM scripts:

```bash
cd slurm_jobs
sbatch run_centrality.slurm
sbatch run_communities.slurm
sbatch run_link_prediction.slurm
# Or submit all with dependencies
bash submit_all.sh
```

## 📚 References

### Textbooks
- **Mining of Massive Datasets** (3rd Edition)
  - Chapter 5: Link Analysis
  - Section 5.4: Link Spam (Spam Farms, TrustRank, Spam Mass)

### Datasets
- **Amazon Product Co-Purchasing Network**: [SNAP Dataset](https://snap.stanford.edu/data/bigdata/communities/com-amazon.ungraph.txt.gz)

### Libraries
- [NetworkX](https://networkx.org/) - Network analysis
- [scikit-learn](https://scikit-learn.org/) - Machine learning
- [pandas](https://pandas.pydata.org/) - Data manipulation
- [matplotlib](https://matplotlib.org/) - Visualization

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Atharv Raotole**
- GitHub: [@AtharvRaotole](https://github.com/AtharvRaotole)
- Project: [Amazon-network-analysis](https://github.com/AtharvRaotole/Amazon-network-analysis)

## 🙏 Acknowledgments

- Stanford SNAP for providing the Amazon dataset
- NetworkX community for excellent documentation
- Textbook authors for algorithm descriptions

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**⭐ If you find this project useful, please consider giving it a star!**
