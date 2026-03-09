FROM continuumio/miniconda3:23.5.2-0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create isolated Python 3.11 environment for ambertools (cpptraj bundled inside)
RUN conda create -n amber_env python=3.11 -c conda-forge ambertools -y && conda clean -afy

# Set working directory
WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies into base env
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# HuggingFace Spaces runs as non-root user 1000
RUN useradd -m -u 1000 user && chown -R user:user /app
USER user

# Expose port 7860 (required by HuggingFace Spaces)
EXPOSE 7860

ENV PORT=7860
ENV CPPTRAJ_PATH=/opt/conda/envs/amber_env/bin/cpptraj
ENV FLASK_SECRET_KEY=cpptrajgpt-hf-spaces-secret

CMD ["python", "server.py"]
