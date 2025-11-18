#!/bin/bash
#SBATCH --job-name=link_prediction
#SBATCH --output=results/logs/link_prediction_%j.out
#SBATCH --error=results/logs/link_prediction_%j.err
#SBATCH --time=12:00:00
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

# Run link prediction
echo "Starting link prediction analysis..."
python -c "
import sys
import pickle
sys.path.insert(0, 'src')
from link_prediction import evaluate_link_prediction
from ml_link_prediction import prepare_training_data, train_random_forest, evaluate_ml_model
from data_loader import load_saved_graph

# Load training graph and test edges
with open('data/processed/splits/train_graph.pkl', 'rb') as f:
    G_train = pickle.load(f)

with open('data/processed/splits/positive_test_edges.pkl', 'rb') as f:
    pos_test = pickle.load(f)

with open('data/processed/splits/negative_test_edges.pkl', 'rb') as f:
    neg_test = pickle.load(f)

print(f'Training graph: {G_train.number_of_edges()} edges')
print(f'Test edges: {len(pos_test)} positive + {len(neg_test)} negative')

# Evaluate similarity methods
print('Evaluating similarity methods...')
methods = ['common_neighbors', 'adamic_adar', 'jaccard']
for method in methods:
    metrics = evaluate_link_prediction(G_train, pos_test[:1000], neg_test[:1000], method=method)
    print(f'{method}: F1={metrics[\"f1\"]:.4f}, AUC-ROC={metrics[\"auc_roc\"]:.4f}')

# Train and evaluate ML model
print('Training ML model...')
X_train, y_train = prepare_training_data(G_train, pos_test[:2000], neg_test[:2000])
model, scaler = train_random_forest(X_train, y_train, n_estimators=100, random_state=42)
ml_metrics, _ = evaluate_ml_model(model, G_train, pos_test[:1000], neg_test[:1000], scaler=scaler)
print(f'ML Model: F1={ml_metrics[\"f1\"]:.4f}, AUC-ROC={ml_metrics[\"auc_roc\"]:.4f}')

print('Link prediction analysis complete!')
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

