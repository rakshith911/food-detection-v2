# Nutrition Video Analysis - ECS Worker Container
# CPU-optimized for Fargate (no GPU support)

# Use Amazon ECR Public Gallery to avoid Docker Hub rate limits
FROM public.ecr.aws/docker/library/python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HUB_OFFLINE=0
ENV TRANSFORMERS_OFFLINE=0
ENV TRUST_REMOTE_CODE=1
ENV HF_HOME=/app/hf_cache
ENV TRANSFORMERS_CACHE=/app/hf_cache/hub
ENV TMPDIR=/tmp
ENV TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_app
ENV USER=appuser
ENV LOGNAME=appuser
ENV ZOEDEPTH_REPO_DIR=/app/vendor/ZoeDepth

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

RUN mkdir -p /app/hf_cache /app/vendor /app/PRODUCTION/model_assets /tmp/torchinductor_app

# CRITICAL: Install NumPy 1.24.3 FIRST before PyTorch
RUN pip install --no-cache-dir "numpy==1.24.3"

# Install PyTorch CPU version (will use the pinned NumPy 1.24.3)
RUN pip install --no-cache-dir \
    torch==2.4.1 torchvision==0.19.1

# Copy and install Python dependencies
COPY requirements.txt .

# CRITICAL: Install requirements WITHOUT sentence-transformers first
# sentence-transformers will be installed separately with modern huggingface_hub
RUN grep -v "sentence-transformers" requirements.txt > /tmp/requirements_no_st.txt && \
    echo "numpy==1.24.3" > /tmp/constraints.txt && \
    pip install --no-cache-dir -r /tmp/requirements_no_st.txt -c /tmp/constraints.txt && \
    pip install --no-cache-dir --force-reinstall --no-deps "numpy==1.24.3" && \
    echo "✅ Installed all packages except sentence-transformers"

# CRITICAL: Install sentence-transformers 3.3.1 which uses modern huggingface_hub (no cached_download)
# Version 3.3.1 uses hf_hub_download instead of the deprecated cached_download
RUN echo "numpy==1.24.3" > /tmp/constraints_st.txt && \
    pip install --no-cache-dir "sentence-transformers==3.3.1" -c /tmp/constraints_st.txt && \
    pip check && \
    python -c "import numpy; actual_version = numpy.__version__; assert actual_version == '1.24.3', f'CRITICAL: NumPy version {actual_version} is wrong! Expected 1.24.3. Build will fail.'" && \
    python -c "from sentence_transformers import SentenceTransformer; print('✅ sentence-transformers 3.3.1 imported successfully')" && \
    echo "✅ Verified NumPy 1.24.3 and sentence-transformers 3.3.1 are installed"

# Pin transformers to a known stable version — git HEAD broke CLIPModel CPU init.
# CRITICAL: Reinstall NumPy after transformers to prevent upgrade
RUN pip install --no-cache-dir --force-reinstall "transformers==4.47.1" && \
    pip install --no-cache-dir --force-reinstall --no-deps "numpy==1.24.3" && \
    python -c "import numpy; assert numpy.__version__ == '1.24.3', f'NumPy upgraded to {numpy.__version__} after transformers install!'"

# Create dummy flash_attn module to satisfy Florence-2 import check
# The actual model will use eager attention as specified in attn_implementation parameter
RUN mkdir -p /usr/local/lib/python3.10/site-packages/flash_attn && \
    echo "# Dummy flash_attn module for Florence-2 CPU compatibility" > /usr/local/lib/python3.10/site-packages/flash_attn/__init__.py

# Patch transformers import_utils.py to handle missing flash_attn in PACKAGE_DISTRIBUTION_MAPPING.
# transformers git HEAD (e1b80de) calls PACKAGE_DISTRIBUTION_MAPPING["flash_attn"] which KeyErrors
# on CPU containers where flash_attn is not a registered package. Change to .get() so it returns
# False (unavailable) instead of crashing. This keeps eager attention as the fallback.
RUN IMPORT_UTILS="/usr/local/lib/python3.10/site-packages/transformers/utils/import_utils.py" && \
    sed -i 's/PACKAGE_DISTRIBUTION_MAPPING\["flash_attn"\]/PACKAGE_DISTRIBUTION_MAPPING.get("flash_attn", [])/g' "$IMPORT_UTILS" && \
    python -c "from transformers import AutoModel, AutoProcessor; print('✅ transformers imports OK after flash_attn patch')"

# Install ZoeDepth source package from the official repository so the packaged
# local checkpoint in PRODUCTION can be loaded in the same runtime.
RUN git clone --depth 1 https://github.com/isl-org/ZoeDepth.git /app/vendor/ZoeDepth && \
    python -c "import sys; sys.path.insert(0, '/app/vendor/ZoeDepth'); import zoedepth; print('✅ zoedepth imported successfully')"

# Copy worker script
COPY worker.py .

# Copy AI processing modules
COPY app/ ./app/

# Clear Python cache AFTER copying to ensure fresh code (no cached bytecode)
RUN find /app -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
    find /app -type f -name "*.pyc" -delete 2>/dev/null || true && \
    find /app -type f -name "*.pyo" -delete 2>/dev/null || true && \
    python3 -Bc "import compileall; compileall.compile_dir('/app', force=True)" 2>/dev/null || true && \
    echo "Python cache cleared"

# Copy SAM2 package to root so it's importable as 'sam2'
COPY sam2_package/sam2 ./sam2/
COPY sam2_package/checkpoints ./checkpoints/

# Copy nutrition RAG system
COPY nutrition_rag_system.py .

# Verify all files are in place AND confirm new video pipeline is deployed
RUN ls -la /app/ && \
    test -f /app/worker.py && \
    test -f /app/nutrition_rag_system.py && \
    python3 -c "import sys; src=open('/app/app/pipeline.py').read(); assert '_load_all_video_frames' in src, 'FATAL: new video pipeline not deployed'; assert src.count('def process_video') >= 1, 'FATAL: process_video missing'; print('pipeline.py OK')" && \
    grep -c "_load_all_video_frames" /app/app/pipeline.py && \
    echo "All required files present"

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python3 -c "print('healthy')" || exit 1

# Run worker
CMD ["python3", "worker.py"]
