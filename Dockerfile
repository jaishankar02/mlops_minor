# 1. Base Image: PyTorch with CUDA 12.1 support
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# 2. Environment Setup
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# 3. Install System Dependencies
# These are required for image processing libraries to function in Linux
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Install Python Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. Copy Project Files
# We copy everything in the current directory to /app
COPY . .

# 6. Set User Permissions (Good practice for security)
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# 7. Default Command
# This runs the training script by default
CMD ["python", "train.py"]
