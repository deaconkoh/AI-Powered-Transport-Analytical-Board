# Base Python image (lightweight)
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Environment settings (no bytecode, faster pip)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System dependencies for PyTorch + scientific packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Copy dependency list first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip setuptools wheel \
 && pip install -r requirements.txt

# Copy the entire project into the container
COPY . .

# Expose the port your app listens on
EXPOSE 8080

# Use Gunicorn to run Flask app in production mode
CMD ["gunicorn", "--workers", "2", "--threads", "4", "-b", "0.0.0.0:8080", "server:app"]
