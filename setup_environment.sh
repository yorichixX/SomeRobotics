#!/usr/bin/env bash
# Quick environment setup script for Ubuntu / Linux 4090 Workstation

set -e

echo "=== Setting up Eric Robots Environment on Linux Workstation ==="

# Check Python version
python3 --version

# Create virtual environment if not already active
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Creating virtual environment .venv..."
    python3 -m venv .venv
    source .venv/bin/activate
fi

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA 12.1+ support
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install requirements
echo "Installing remaining packages..."
pip install -r requirements.txt

# Verify CUDA
python3 -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo "=== Setup Complete! ==="
