FROM python:3.10-slim

# Install System Dependencies for OpenCV Headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Requirements and Install Lightweight CPU Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy Project Files
COPY . .

# Expose Port
EXPOSE 10000

# Start FastAPI Application with Uvicorn
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "10000"]
