# Tests Directory

This directory contains verification and testing scripts for the project.

## verify_setup.py

Comprehensive verification script that tests all Module 1 functionality:

1. ✅ Checks if all required files exist
2. ✅ Tests data loading (graph and communities)
3. ✅ Tests preprocessing functions
4. ✅ Tests train/test split creation
5. ✅ Tests exploratory analysis functions
6. ✅ Final validation

### Usage

```bash
# From project root
python tests/verify_setup.py
```

Or make it executable and run directly:

```bash
chmod +x tests/verify_setup.py
./tests/verify_setup.py
```

### Requirements

- All dataset files in `data/raw/`
- All source modules in `src/`
- Python dependencies installed

### Output

The script provides colored output (if colorama is installed):
- ✅ Green for success
- ❌ Red for errors
- ℹ️  Cyan for info

Exit code 0 on success, 1 on failure.

