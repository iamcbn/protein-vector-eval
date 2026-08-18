# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
.venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch for CUDA (ensure it works on RTX 4000, usually CUDA 11.8 or 12.1 are fine, using 12.1 as default for recent torch versions)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
pip install -r requirements.txt

Write-Host "Environment setup complete!" -ForegroundColor Green
