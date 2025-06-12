# Use Python 3.9 slim as base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT 8080
ENV COMMANDLINE_ARGS="--no-gradio-queue"
# Install system dependencies
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create requirements.txt with all necessary dependencies
RUN echo "torch>=1.9.0" > requirements.txt && \
    echo "torchvision>=0.10.0" >> requirements.txt && \
    echo "gradio>=3.0.0" >> requirements.txt && \
    echo "opencv-python>=4.5.0" >> requirements.txt && \
    echo "numpy>=1.21.0" >> requirements.txt && \
    echo "pandas>=1.3.0" >> requirements.txt && \
    echo "Pillow>=8.3.0" >> requirements.txt && \
    echo "tqdm>=4.62.0" >> requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY gradio_app_v2.py /app/
COPY main_driver/ /app/main_driver/

# Create necessary directories
RUN mkdir -p /app/logs /app/tmp

# Expose port 8080 (as used in the Gradio app)
EXPOSE 8080

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8080/ || exit 1

# Set up proper permissions
RUN chmod +x /app/gradio_app_v2.py

# Run the application
CMD ["python", "gradio_app_v2.py"] 