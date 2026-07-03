FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies needed for OpenCV
RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

# Install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy models and application
COPY models /models
COPY app /app/app

# Set user for Hugging Face Spaces compatibility
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /app

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
