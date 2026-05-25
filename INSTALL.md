# Setup

## 1. Create virtual environment

python -m venv odvenv

## 2. Activate

Windows:
odvenv\Scripts\activate

## 3. Install CUDA-enabled PyTorch, CUDA-enabled onnxruntime

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

pip install onnxruntime-gpu

## 4. Install remaining dependencies

pip install -r requirements.txt


for onnx if forgot

pip uninstall onnxruntime
pip install onnxruntime-gpu