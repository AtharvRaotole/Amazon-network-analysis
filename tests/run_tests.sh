#!/bin/bash
# Script to run all tests with coverage

echo "=========================================="
echo "Running Unit Tests"
echo "=========================================="

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "⚠️  pytest not found. Installing..."
    pip install pytest pytest-cov
fi

# Run tests
echo ""
echo "Running all tests..."
pytest tests/test_modules.py -v --cov=src --cov-report=html --cov-report=term

echo ""
echo "=========================================="
echo "Test Results Summary"
echo "=========================================="
echo "Coverage report: htmlcov/index.html"
echo "=========================================="

