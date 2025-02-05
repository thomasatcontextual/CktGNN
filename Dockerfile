# Use Python 3.9 slim as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Create a volume for pip cache
VOLUME /root/.cache/pip

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libjpeg-dev \
    libpng-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create cache directories for wheels and build
RUN mkdir -p /root/.cache/pip/wheels

# Copy requirements and validation files first (for better caching)
COPY requirements.txt env_validation.py ./

# Install Python packages in the correct order (as we discovered works)
RUN --mount=type=cache,target=/root/.cache/pip,id=pip_cache \
    --mount=type=cache,target=/root/.cache/pip/wheels,id=pip_wheels \
    TORCH_CPU=1 pip install --no-cache-dir torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir torch-geometric==2.3.1 \
    && pip install --no-cache-dir torch-scatter==2.1.1 torch-sparse==0.6.17 torch-cluster==1.6.1 \
    && pip install --no-cache-dir matplotlib \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create directories for data and results
RUN mkdir -p OCB/CktBench101 results

# Run validation on build to verify environment
RUN python env_validation.py

# Default command to show help
CMD ["python", "main.py", "--help"] 