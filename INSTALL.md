# Setup

## 1. Create virtual environment

python -m venv odvenv

## 2. Activate

Windows:
odvenv\Scripts\activate

## 3. Install CUDA-enabled PyTorch

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

## 4. Install remaining dependencies

pip install -r requirements.txt