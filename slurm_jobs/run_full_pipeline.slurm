#!/bin/bash
#SBATCH --job-name=full_pipeline
#SBATCH --output=results/logs/full_pipeline_%j.out
#SBATCH --error=results/logs/full_pipeline_%j.err
#SBATCH --time=24:00:00
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

# Create logs directory
mkdir -p results/logs

# Run full pipeline
echo "=========================================="
echo "Running Full Analysis Pipeline"
echo "=========================================="

# Step 1: Data Loading and Preprocessing
echo ""
echo "Step 1: Data Loading and Preprocessing"
echo "----------------------------------------"
python -c "
import sys
sys.path.insert(0, 'src')
from data_loader import download_dataset, load_graph, save_graph
from preprocessing import remove_self_loops, get_largest_component, create_train_test_split

# Download and load data
print('Loading graph...')
G = load_graph('data/raw/com-amazon.ungraph.txt.gz')
print(f'Original: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges')

# Preprocessing
G = remove_self_loops(G)
G = get_largest_component(G)
print(f'Cleaned: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges')

# Save cleaned graph
save_graph(G, 'data/processed/amazon_graph_cleaned.pkl')
print('Cleaned graph saved')

# Create train/test split
G_train, pos_test, neg_test = create_train_test_split(G, test_ratio=0.2, seed=42)
print('Train/test split created')
"

if [ $? -ne 0 ]; then
    echo "ERROR: Data loading/preprocessing failed"
    exit 1
fi

# Step 2: Centrality Analysis
echo ""
echo "Step 2: Centrality Analysis"
echo "----------------------------------------"
python -c "
import sys
sys.path.insert(0, 'src')
from centrality_analysis import compare_centrality_measures
from data_loader import load_saved_graph

G = load_saved_graph('data/processed/amazon_graph_cleaned.pkl')
results = compare_centrality_measures(G, k=100)
print('Centrality analysis complete')
"

if [ $? -ne 0 ]; then
    echo "ERROR: Centrality analysis failed"
    exit 1
fi

# Step 3: Community Detection
echo ""
echo "Step 3: Community Detection"
echo "----------------------------------------"
python -c "
import sys
sys.path.insert(0, 'src')
from community_detection import run_all_community_detection
from data_loader import load_saved_graph

G = load_saved_graph('data/processed/amazon_graph_cleaned.pkl')
results = run_all_community_detection(G, louvain_resolution=1.0, seed=42)
print('Community detection complete')
"

if [ $? -ne 0 ]; then
    echo "ERROR: Community detection failed"
    exit 1
fi

# Step 4: Link Prediction
echo ""
echo "Step 4: Link Prediction"
echo "----------------------------------------"
python -c "
import sys
import pickle
sys.path.insert(0, 'src')
from link_prediction import evaluate_link_prediction
from ml_link_prediction import prepare_training_data, train_random_forest, evaluate_ml_model

with open('data/processed/splits/train_graph.pkl', 'rb') as f:
    G_train = pickle.load(f)

with open('data/processed/splits/positive_test_edges.pkl', 'rb') as f:
    pos_test = pickle.load(f)

with open('data/processed/splits/negative_test_edges.pkl', 'rb') as f:
    neg_test = pickle.load(f)

# Evaluate similarity methods
for method in ['common_neighbors', 'adamic_adar', 'jaccard']:
    metrics = evaluate_link_prediction(G_train, pos_test[:1000], neg_test[:1000], method=method)
    print(f'{method}: F1={metrics[\"f1\"]:.4f}')

# Train ML model
X_train, y_train = prepare_training_data(G_train, pos_test[:2000], neg_test[:2000])
model, scaler = train_random_forest(X_train, y_train, n_estimators=100, random_state=42)
ml_metrics, _ = evaluate_ml_model(model, G_train, pos_test[:1000], neg_test[:1000], scaler=scaler)
print(f'ML Model: F1={ml_metrics[\"f1\"]:.4f}')

print('Link prediction complete')
"

if [ $? -ne 0 ]; then
    echo "ERROR: Link prediction failed"
    exit 1
fi

# Final summary
echo ""
echo "=========================================="
echo "Full Pipeline Completed Successfully"
echo "End Time: $(date)"
echo "=========================================="
exit 0

