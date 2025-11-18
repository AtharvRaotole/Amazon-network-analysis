@echo off
REM Setup script for Amazon Co-Purchasing Network project
REM For Windows systems

echo ======================================
echo 🚀 Amazon Network Analysis Setup
echo ======================================
echo.

REM Step 1: Create data directories
echo [1/7] Creating data directories...
if not exist "data\raw" mkdir data\raw
if not exist "data\processed" mkdir data\processed
if not exist "results\figures" mkdir results\figures
if not exist "results\tables" mkdir results\tables
if not exist "results\models" mkdir results\models
echo ✅ Directories created

REM Step 2: Download dataset files
echo.
echo [2/7] Downloading dataset files...

set GRAPH_URL=https://snap.stanford.edu/data/bigdata/communities/com-amazon.ungraph.txt.gz
set COMMUNITIES_URL=https://snap.stanford.edu/data/bigdata/communities/com-amazon.all.cmty.txt.gz

if not exist "data\raw\com-amazon.ungraph.txt.gz" (
    echo ℹ️  Downloading graph file...
    powershell -Command "Invoke-WebRequest -Uri '%GRAPH_URL%' -OutFile 'data\raw\com-amazon.ungraph.txt.gz'"
    echo ✅ Graph file downloaded
) else (
    echo ℹ️  Graph file already exists, skipping download
)

if not exist "data\raw\com-amazon.all.cmty.txt.gz" (
    echo ℹ️  Downloading communities file...
    powershell -Command "Invoke-WebRequest -Uri '%COMMUNITIES_URL%' -OutFile 'data\raw\com-amazon.all.cmty.txt.gz'"
    echo ✅ Communities file downloaded
) else (
    echo ℹ️  Communities file already exists, skipping download
)

REM Step 3: Check dataset files
echo.
echo [3/7] Checking dataset files...
if exist "data\raw\com-amazon.ungraph.txt.gz" if exist "data\raw\com-amazon.all.cmty.txt.gz" (
    echo ✅ All dataset files present
    echo ℹ️  Note: Files are kept in .gz format (loader handles decompression)
) else (
    echo ❌ Some dataset files are missing
    exit /b 1
)

REM Step 4: Create virtual environment
echo.
echo [4/7] Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo ✅ Virtual environment created
) else (
    echo ℹ️  Virtual environment already exists
)

REM Step 5: Activate virtual environment and install requirements
echo.
echo [5/7] Installing requirements...
call venv\Scripts\activate.bat

REM Upgrade pip first
python -m pip install --upgrade pip --quiet

REM Install requirements
if exist "requirements.txt" (
    pip install -r requirements.txt
    echo ✅ Requirements installed
) else (
    echo ❌ requirements.txt not found
    exit /b 1
)

REM Install colorama for colored output
pip install colorama --quiet

REM Install ipykernel for Jupyter
echo.
echo ℹ️  Installing Jupyter kernel...
pip install ipykernel --quiet
python -m ipykernel install --user --name=amazon-analysis --display-name "Python (amazon-analysis)"
echo ✅ Jupyter kernel installed

REM Step 6: Run verification script
echo.
echo [6/7] Running verification script...
if exist "tests\verify_setup.py" (
    python tests\verify_setup.py
    if errorlevel 1 (
        echo ❌ Verification failed
        exit /b 1
    ) else (
        echo ✅ Verification passed
    )
) else (
    echo ❌ Verification script not found
    exit /b 1
)

REM Step 7: Success message
echo.
echo [7/7] Setup complete!
echo.
echo ======================================
echo ✅ SETUP COMPLETED SUCCESSFULLY!
echo ======================================
echo.
echo Next steps:
echo 1. Activate virtual environment: venv\Scripts\activate
echo 2. Start Jupyter: jupyter notebook
echo 3. Open notebooks\01_data_exploration.ipynb
echo.
echo To deactivate virtual environment: deactivate
echo.

pause

