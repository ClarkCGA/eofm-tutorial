# GPU-capable PyTorch base image
FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime

# Workdir inside container
WORKDIR /app

# OS deps (adjust as needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements & install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and configs
COPY src ./src
COPY configs ./configs
COPY diffusion ./diffusion
COPY run_it.py ./run_it.py 
COPY README*.md ./

# Ensure /app is on PYTHONPATH
ENV PYTHONPATH=/app

# Default entrypoint (change src/train.py if your main script has a different name)
ENTRYPOINT ["python" , "run_it.py"]
