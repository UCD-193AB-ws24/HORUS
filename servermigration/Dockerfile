FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    python3-dev \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY *.py ./
COPY gloss.csv ./

# Create model directory and download necessary models
RUN mkdir -p /app/models

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PRELOAD_MODELS=false

# Verify CUDA and NVCC availability
RUN nvcc --version && python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Set command to run the handler
CMD ["python", "handler.py"]