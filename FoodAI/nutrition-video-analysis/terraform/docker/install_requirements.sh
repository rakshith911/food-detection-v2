#!/bin/bash
# Install requirements for local pipeline execution
# This script handles the installation step by step to avoid timeouts

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Installing Pipeline Requirements"
echo "=========================================="
echo ""

# Activate virtual environment
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Run setup_local.sh first."
    exit 1
fi

source venv/bin/activate

echo "📦 Step 1/4: Installing NumPy (already done if you see this)..."
pip install "numpy==1.24.3" --quiet

echo "📦 Step 2/4: Installing PyTorch (this is LARGE ~500MB, will take 5-10 minutes)..."
echo "   ⏳ Please be patient - downloading PyTorch..."
pip install torch==2.4.1 torchvision==0.19.1

echo "📦 Step 3/4: Installing core dependencies..."
pip install \
    "git+https://github.com/huggingface/transformers.git" \
    opencv-python-headless==4.8.1.78 \
    Pillow==10.1.0 \
    scipy==1.11.4 \
    einops==0.7.0 \
    timm==0.9.12 \
    h5py==3.10.0 \
    "huggingface-hub>=0.34.0" \
    "httpx>=0.28.1" \
    "accelerate>=1.10.0"

echo "📦 Step 4/5: Installing remaining dependencies (excluding sentence-transformers)..."
grep -v "sentence-transformers" requirements.txt > /tmp/requirements_no_st_local.txt
pip install -r /tmp/requirements_no_st_local.txt

echo "📦 Step 5/5: Installing sentence-transformers separately for transformer compatibility..."
pip install "sentence-transformers==3.3.1" --no-deps
pip install "scikit-learn==1.5.2"

echo "📦 Step 6/6: Installing ZoeDepth source package..."
if [ ! -d "vendor/ZoeDepth" ]; then
    mkdir -p vendor
    git clone --depth 1 https://github.com/isl-org/ZoeDepth.git vendor/ZoeDepth
fi
echo "✅ ZoeDepth source cloned to vendor/ZoeDepth (loaded via PYTHONPATH/runtime path injection)"

echo ""
echo "=========================================="
echo "✅ Installation Complete!"
echo "=========================================="
echo ""
echo "To test the pipeline:"
echo "  source venv/bin/activate"
echo "  python3 test_worker_simple.py /path/to/your/image.jpg"
echo ""
