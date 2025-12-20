# Use NVIDIA CUDA base image with cuDNN
FROM nvidia/cuda:12.4.1-base-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
# git: for installing dependencies from git
# ffmpeg: required for audio processing
# libsndfile1: required for torchaudio
RUN apt-get update && apt-get install -y nala && nala install -y \
    python3.10 \
    python3-pip \
    python3-venv \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
RUN pip3 install uv

# Set working directory
WORKDIR /app

# Copy project files
# Copy project definition
COPY pyproject.toml README.md ./

# Create virtual environment and install dependencies
# We use --system to install into the container's python environment directly or use uv's venv
RUN uv venv .venv
ENV PATH="/app/.venv/bin:$PATH"

# Create dummy package structure to allow installing dependencies without copying full source
# This ensures cached dependencies aren't invalidated when source code changes
RUN mkdir whisperx && touch whisperx/__init__.py && \
    uv pip install . && \
    rm -rf whisperx

# Add PyTorch's bundled libraries to LD_LIBRARY_PATH so ctranslate2 can find them
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/app/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:/app/.venv/lib/python3.10/site-packages/nvidia/cublas/lib"

# Copy actual project files (this layer changes more often)
COPY transcribe_with_diarization.py .
COPY whisperx whisperx

# Entrypoint
ENTRYPOINT ["python3", "transcribe_with_diarization.py"]
