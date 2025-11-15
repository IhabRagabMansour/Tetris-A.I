# Installation Guide

## Prerequisites
- Python 3.8 or higher (tested with Python 3.11.14)
- NVIDIA GPU with CUDA support (optional, for GPU acceleration)
- CUDA 11.8 toolkit installed (if using GPU)

## Quick Start

### 1. Create Virtual Environment (Recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

#### Option A: GPU Support (CUDA 11.8)
For systems with NVIDIA GPU and CUDA 11.8:

```bash
# Install PyTorch with CUDA 11.8 support
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install -r requirements.txt
```

#### Option B: CPU Only
For systems without GPU or CUDA:

```bash
# Install all dependencies (PyTorch will default to CPU version)
pip install -r requirements.txt
```

#### Option C: Different CUDA Version
If you have a different CUDA version:

**CUDA 12.4:**
```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

**CUDA 12.6:**
```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

### 3. Verify Installation

Run this command to verify everything is installed correctly:

```bash
python -c "import torch; import numba; import numpy; print(f'PyTorch: {torch.__version__}'); print(f'NumPy: {numpy.__version__}'); print(f'Numba: {numba.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch: 2.6.0+cu118
NumPy: 2.2.3
Numba: 0.61.2
CUDA Available: True  (or False if CPU-only)
```

## Package Compatibility Notes

### Fixed Compatibility Issues

The original `requirements.txt` had an incompatibility:
- ❌ `numba==0.61.0` + `numpy==2.2.3` → **INCOMPATIBLE**
- ✅ `numba==0.61.2` + `numpy==2.2.3` → **COMPATIBLE**

**Changes made:**
- Updated `numba` from 0.61.0 to 0.61.2 (adds NumPy 2.2 support)
- Removed `+cu118` suffix from torch version (handled via pip index-url)

### Why PyTorch Needs Special Installation

PyTorch CUDA variants are not available on PyPI. They must be installed from PyTorch's custom package index using the `--index-url` flag. This is why we install PyTorch separately.

## Troubleshooting

### Issue: "Numba needs NumPy 2.1 or less"
**Solution:** Make sure you're using `numba>=0.61.2`. Update with:
```bash
pip install --upgrade numba==0.61.2
```

### Issue: "Could not find a version that satisfies the requirement torch"
**Solution:** Use the correct PyTorch index URL:
```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118
```

### Issue: CUDA not available even though GPU is present
**Solutions:**
1. Verify CUDA toolkit is installed: `nvcc --version`
2. Check GPU is detected: `nvidia-smi`
3. Reinstall PyTorch with correct CUDA version
4. Verify with: `python -c "import torch; print(torch.cuda.is_available())"`

### Issue: Import errors or version conflicts
**Solution:** Create a fresh virtual environment:
```bash
deactivate  # if already in a venv
rm -rf venv
python3 -m venv venv
source venv/bin/activate
# Then follow installation steps again
```

## Running the Project

After successful installation:

### Version 1 (Multiprocessing approach)
```bash
cd Version1
python -c "import train; train.run_game(True)"   # Enable slow drop
python -c "import train; train.run_game(False)"  # Disable slow drop
```

### Version 2 (Optimized single-process with Genetic Algorithm)
```bash
cd Version2
python genetic_algo.py
```

## System Requirements

**Minimum:**
- CPU: Multi-core processor (4+ cores recommended)
- RAM: 4GB
- Storage: 1GB free space

**Recommended for GPU training:**
- CPU: 8+ cores
- RAM: 8GB+
- GPU: NVIDIA GPU with 4GB+ VRAM
- CUDA: 11.8 or higher

## Additional Resources

- PyTorch Installation Guide: https://pytorch.org/get-started/locally/
- Numba Documentation: https://numba.readthedocs.io/
- Project README: `README.md`
