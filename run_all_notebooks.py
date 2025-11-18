"""
Execute all notebooks to generate complete results.

This script runs all notebook code cells to generate outputs and results.
"""

import os
import sys
import json
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path

def run_notebook(notebook_path):
    """Execute a Jupyter notebook."""
    print(f"\n{'='*60}")
    print(f"Running: {notebook_path}")
    print(f"{'='*60}\n")
    
    try:
        # Read notebook
        with open(notebook_path, 'r') as f:
            nb = nbformat.read(f, as_version=4)
        
        # Execute
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        ep.preprocess(nb, {'metadata': {'path': str(Path(notebook_path).parent)}})
        
        # Save executed notebook
        output_path = notebook_path.replace('.ipynb', '_executed.ipynb')
        with open(output_path, 'w') as f:
            nbformat.write(nb, f)
        
        print(f"✅ {notebook_path} executed successfully")
        print(f"   Output saved to: {output_path}")
        return True
    
    except Exception as e:
        print(f"❌ {notebook_path} failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all notebooks."""
    notebooks = [
        'notebooks/01_data_exploration.ipynb',
        'notebooks/02_centrality_analysis.ipynb',
        'notebooks/03_community_detection.ipynb',
        'notebooks/04_link_prediction.ipynb',
        'notebooks/05_spam_farm_analysis.ipynb',
    ]
    
    results = []
    for nb_path in notebooks:
        if os.path.exists(nb_path):
            success = run_notebook(nb_path)
            results.append((nb_path, success))
        else:
            print(f"⚠️  {nb_path} not found, skipping")
            results.append((nb_path, None))
    
    # Summary
    print(f"\n{'='*60}")
    print("NOTEBOOK EXECUTION SUMMARY")
    print(f"{'='*60}\n")
    
    for nb_path, success in results:
        if success is True:
            print(f"✅ {nb_path}")
        elif success is False:
            print(f"❌ {nb_path}")
        else:
            print(f"⚠️  {nb_path} (not found)")
    
    passed = sum(1 for _, s in results if s is True)
    total = sum(1 for _, s in results if s is not None)
    
    print(f"\n✅ {passed}/{total} notebooks executed successfully")

if __name__ == "__main__":
    main()

