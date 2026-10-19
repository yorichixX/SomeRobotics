@echo off
echo === Setting up Eric Robots Environment on Windows Workstation ===

python --version

if not defined VIRTUAL_ENV (
    echo Creating virtual environment .venv...
    python -m venv .venv
    call .venv\Scripts\activate.bat
)

python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt

python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
echo === Setup Complete! ===
