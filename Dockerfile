# Use Python 3.13 slim as base (matches your local env more closely)
FROM python:3.13-slim

# Set working directory
WORKDIR /usr/src/app

# Install necessary system dependencies for OpenCV and Compilers
# This fixes the "ImportError" and "Build Timeout" issues
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential gfortran \
    libsm6 libxext6 libxrender1 libfontconfig1 libice6 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Environment variable for the port Render requires
ENV PORT 10000

# Expose the port
EXPOSE 10000

# Start the application
CMD python -m uvicorn main:app --host 0.0.0.0 --port $PORT
