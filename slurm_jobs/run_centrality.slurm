#!/bin/bash
#SBATCH --job-name=centrality_analysis
#SBATCH --output=results/logs/centrality_%j.out
#SBATCH --error=results/logs/centrality_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your_email@example.com

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Load Python module (adjust based on your HPC cluster)
# module load python/3.9
# module load gcc/9.3.0

# Activate virtual environment
source venv/bin/activate

# Set working directory
cd $SLURM_SUBMIT_DIR

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run centrality analysis
echo "Starting centrality analysis..."
python -c "
import sys
sys.path.insert(0, 'src')
from centrality_analysis import compare_centrality_measures
from data_loader import load_saved_graph
import networkx as nx

# Load graph
G = load_saved_graph('data/processed/amazon_graph_cleaned.pkl')
print(f'Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges')

# Run centrality analysis
results = compare_centrality_measures(G, k=100)
print('Centrality analysis complete!')
"

# Check exit status
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "Job completed successfully"
    echo "End Time: $(date)"
    echo "=========================================="
    exit 0
else
    echo "=========================================="
    echo "Job failed with exit code $?"
    echo "End Time: $(date)"
    echo "=========================================="
    exit 1
fi

