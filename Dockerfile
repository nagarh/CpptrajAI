FROM condaforge/mambaforge:latest

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install cpptraj in its own isolated conda environment (avoids Python 3.12 pin conflict)
RUN mamba create -n cpptraj_env -c conda-forge -c bioconda cpptraj -y && mamba clean -afy

# Set working directory
WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies into the base (Python 3.12) environment
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# HuggingFace Spaces runs as non-root user 1000
RUN useradd -m -u 1000 user && chown -R user:user /app
USER user

# Expose port 7860 (required by HuggingFace Spaces)
EXPOSE 7860

ENV PORT=7860
ENV CPPTRAJ_PATH=/opt/conda/envs/cpptraj_env/bin/cpptraj
ENV FLASK_SECRET_KEY=cpptrajgpt-hf-spaces-secret

CMD ["python", "server.py"]
