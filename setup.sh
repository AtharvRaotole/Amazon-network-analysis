#!/bin/bash
# Setup script for Amazon Co-Purchasing Network project
# For Linux/Mac systems

set -e  # Exit on error

echo "======================================"
echo "🚀 Amazon Network Analysis Setup"
echo "======================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored messages
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Step 1: Create data/raw directory
echo "[1/7] Creating data directories..."
mkdir -p data/raw data/processed results/figures results/tables results/models
print_success "Directories created"

# Step 2: Download dataset files
echo ""
echo "[2/7] Downloading dataset files..."

GRAPH_URL="https://snap.stanford.edu/data/bigdata/communities/com-amazon.ungraph.txt.gz"
COMMUNITIES_URL="https://snap.stanford.edu/data/bigdata/communities/com-amazon.all.cmty.txt.gz"

if [ ! -f "data/raw/com-amazon.ungraph.txt.gz" ]; then
    print_info "Downloading graph file..."
    if command -v wget &> /dev/null; then
        wget -P data/raw/ "$GRAPH_URL" --progress=bar
    elif command -v curl &> /dev/null; then
        curl -L -o data/raw/com-amazon.ungraph.txt.gz "$GRAPH_URL" --progress-bar
    else
        print_error "Neither wget nor curl found. Please install one of them."
        exit 1
    fi
    print_success "Graph file downloaded"
else
    print_info "Graph file already exists, skipping download"
fi

if [ ! -f "data/raw/com-amazon.all.cmty.txt.gz" ]; then
    print_info "Downloading communities file..."
    if command -v wget &> /dev/null; then
        wget -P data/raw/ "$COMMUNITIES_URL" --progress=bar
    elif command -v curl &> /dev/null; then
        curl -L -o data/raw/com-amazon.all.cmty.txt.gz "$COMMUNITIES_URL" --progress-bar
    else
        print_error "Neither wget nor curl found. Please install one of them."
        exit 1
    fi
    print_success "Communities file downloaded"
else
    print_info "Communities file already exists, skipping download"
fi

# Step 3: Extract .gz files (optional - the loader can handle gzipped files)
echo ""
echo "[3/7] Checking dataset files..."
if [ -f "data/raw/com-amazon.ungraph.txt.gz" ] && [ -f "data/raw/com-amazon.all.cmty.txt.gz" ]; then
    print_success "All dataset files present"
    print_info "Note: Files are kept in .gz format (loader handles decompression)"
else
    print_error "Some dataset files are missing"
    exit 1
fi

# Step 4: Create virtual environment
echo ""
echo "[4/7] Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_info "Virtual environment already exists"
fi

# Step 5: Activate virtual environment and install requirements
echo ""
echo "[5/7] Installing requirements..."
source venv/bin/activate

# Upgrade pip first
pip install --upgrade pip --quiet

# Install requirements
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    print_success "Requirements installed"
else
    print_error "requirements.txt not found"
    exit 1
fi

# Install colorama for colored output in verification script
pip install colorama --quiet

# Install ipykernel for Jupyter
echo ""
print_info "Installing Jupyter kernel..."
pip install ipykernel --quiet
python -m ipykernel install --user --name=amazon-analysis --display-name "Python (amazon-analysis)" || true
print_success "Jupyter kernel installed"

# Step 6: Run verification script
echo ""
echo "[6/7] Running verification script..."
if [ -f "tests/verify_setup.py" ]; then
    python tests/verify_setup.py
    VERIFY_EXIT_CODE=$?
    if [ $VERIFY_EXIT_CODE -eq 0 ]; then
        print_success "Verification passed"
    else
        print_error "Verification failed"
        exit 1
    fi
else
    print_error "Verification script not found"
    exit 1
fi

# Step 7: Success message
echo ""
echo "[7/7] Setup complete!"
echo ""
echo "======================================"
print_success "SETUP COMPLETED SUCCESSFULLY!"
echo "======================================"
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Start Jupyter: jupyter notebook"
echo "3. Open notebooks/01_data_exploration.ipynb"
echo ""
echo "To deactivate virtual environment: deactivate"
echo ""

