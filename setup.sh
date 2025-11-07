#!/bin/bash
# Setup script for SpikeCLIP

echo "=========================================="
echo "SpikeCLIP Setup Script"
echo "=========================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "Virtual environment 'venv' already exists"
    source venv/bin/activate
else
    echo "Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create virtual environment"
        echo "Please install venv: python3 -m ensurepip --upgrade"
        exit 1
    fi
    source venv/bin/activate
    echo "Virtual environment activated"
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p data/U-CALTECH
mkdir -p data/U-CIFAR
mkdir -p models
mkdir -p exp
mkdir -p clip_model
mkdir -p test_outputs

# Upgrade pip
echo "Upgrading pip..."
python3 -m pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
python3 -m pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    echo "Please check requirements.txt and try again"
    exit 1
fi

echo "Dependencies installed successfully!"

# Check CUDA availability
echo "Checking CUDA and PyTorch..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null

if [ $? -ne 0 ]; then
    echo "WARNING: PyTorch not installed or not accessible"
    echo "PyTorch will be installed via requirements.txt"
    echo "Note: CUDA is not available on macOS. PyTorch will use CPU or MPS (Metal Performance Shaders)"
fi

echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "IMPORTANT: Virtual environment is active"
echo "To activate it in future sessions, run:"
echo "  source venv/bin/activate"
echo ""
echo "Next steps:"
echo "1. Download the UHSR dataset from: https://github.com/Evin-X/UHSR"
echo "2. Extract to data/ directory"
echo "3. Run: python3 fetch_data.py --dataset all"
echo "4. Train: python3 train.py --data_type CALTECH"
echo "5. Test: python3 test.py --checkpoint <path_to_checkpoint>"
echo ""
echo "Note: Always activate venv first: source venv/bin/activate"
echo "=========================================="

