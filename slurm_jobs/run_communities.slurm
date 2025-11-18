#!/bin/bash
#SBATCH --job-name=community_detection
#SBATCH --output=results/logs/communities_%j.out
#SBATCH --error=results/logs/communities_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
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

# Run community detection
echo "Starting community detection..."
python -c "
import sys
sys.path.insert(0, 'src')
from community_detection import run_all_community_detection
from data_loader import load_saved_graph

# Load graph
G = load_saved_graph('data/processed/amazon_graph_cleaned.pkl')
print(f'Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges')

# Run community detection
results = run_all_community_detection(G, louvain_resolution=1.0, seed=42)
print('Community detection complete!')
print(f'Louvain: {len(results[\"louvain\"][0])} communities')
print(f'Label Propagation: {len(results[\"label_propagation\"][0])} communities')
print(f'Greedy Modularity: {len(results[\"greedy_modularity\"][0])} communities')
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

