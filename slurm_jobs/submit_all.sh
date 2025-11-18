#!/bin/bash
# Script to submit all SLURM jobs with dependencies
# Usage: ./submit_all.sh

echo "=========================================="
echo "Submitting All SLURM Jobs"
echo "=========================================="

# Create logs directory
mkdir -p results/logs

# Make scripts executable
chmod +x slurm_jobs/*.slurm

# Submit jobs with dependencies
echo ""
echo "1. Submitting centrality analysis job..."
JOB1=$(sbatch slurm_jobs/run_centrality.slurm | awk '{print $4}')
echo "   Job ID: $JOB1"

echo ""
echo "2. Submitting community detection job..."
JOB2=$(sbatch slurm_jobs/run_communities.slurm | awk '{print $4}')
echo "   Job ID: $JOB2"

echo ""
echo "3. Submitting link prediction job..."
JOB3=$(sbatch slurm_jobs/run_link_prediction.slurm | awk '{print $4}')
echo "   Job ID: $JOB3"

echo ""
echo "4. Submitting full pipeline job (runs all sequentially)..."
JOB4=$(sbatch slurm_jobs/run_full_pipeline.slurm | awk '{print $4}')
echo "   Job ID: $JOB4"

echo ""
echo "=========================================="
echo "All Jobs Submitted"
echo "=========================================="
echo "Centrality Analysis:    $JOB1"
echo "Community Detection:    $JOB2"
echo "Link Prediction:        $JOB3"
echo "Full Pipeline:          $JOB4"
echo ""
echo "Check job status with:"
echo "  squeue -u \$USER"
echo ""
echo "View logs in: results/logs/"
echo "=========================================="

