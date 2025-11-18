# Large-Scale Link Analysis: Amazon Product Co-Purchasing Network

## Project Description

This project performs large-scale link analysis on the Amazon Product Co-Purchasing Network dataset from SNAP (Stanford Network Analysis Project). The dataset contains 334,863 nodes and 925,872 edges, representing co-purchasing relationships between Amazon products.

The project aims to:
- Analyze network structure and properties
- Identify communities and clusters
- Perform link prediction and analysis
- Visualize network characteristics
- Extract insights from the co-purchasing patterns

## Installation Instructions

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd finalbda
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Downloading the Dataset

The Amazon co-purchasing network dataset can be downloaded from:
- SNAP Dataset: https://snap.stanford.edu/data/amazon0302.html

Place the raw data files in the `data/raw/` directory.

### Running Analysis

1. **Jupyter Notebooks**: 
   - Navigate to the `notebooks/` directory
   - Launch Jupyter: `jupyter notebook`
   - Open and run the analysis notebooks

2. **Python Scripts**:
   - Scripts are located in the `src/` directory
   - Run from the project root:
   ```bash
   python src/your_script.py
   ```

3. **HPC/SLURM Jobs**:
   - SLURM job scripts are in `slurm_jobs/`
   - Submit jobs using:
   ```bash
   sbatch slurm_jobs/your_job.sh
   ```

## Dataset Information

### Amazon Product Co-Purchasing Network

- **Source**: SNAP (Stanford Network Analysis Project)
- **Nodes**: 334,863 (products)
- **Edges**: 925,872 (co-purchasing relationships)
- **Format**: Edge list or adjacency matrix
- **Description**: Nodes represent products, and edges indicate that two products are frequently co-purchased together

### Dataset Files

- Raw data files should be placed in `data/raw/`
- Processed data will be stored in `data/processed/`
- Analysis results will be saved in `data/results/`

## Project Structure

```
finalbda/
├── data/
│   ├── raw/              # Raw dataset files
│   ├── processed/        # Processed/cleaned data
│   └── results/          # Intermediate analysis results
├── src/                  # Python source code modules
├── notebooks/            # Jupyter notebooks for analysis
├── slurm_jobs/           # HPC/SLURM job scripts
├── results/
│   ├── figures/          # Generated plots and visualizations
│   ├── tables/           # Analysis tables and statistics
│   └── models/           # Trained models and checkpoints
├── requirements.txt      # Python dependencies
├── README.md             # This file
└── .gitignore           # Git ignore rules
```

## Contributing

1. Create a feature branch
2. Make your changes
3. Test thoroughly
4. Submit a pull request

## License

[Specify your license here]

## Contact

[Add contact information or maintainer details]

