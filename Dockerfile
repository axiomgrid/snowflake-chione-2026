# Snowflake Chione 2026 — Docker Environment
FROM python:3.12-slim

WORKDIR /app

# System dependencies: build tools + FFmpeg for H.265 video encoding
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create output directory
RUN mkdir -p results

# Default: run the full 12×12 parameter sweep
CMD ["python3", "-m", "src.cli", "--alpha_min", "0.01", "--alpha_max", "2.5", "--gamma_min", "0.0001", "--gamma_max", "0.01", "--gif"]
