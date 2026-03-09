FROM condaforge/mambaforge:latest

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install cpptraj from conda-forge using mamba (faster solver)
RUN mamba install -y -c conda-forge cpptraj && mamba clean -afy

# Set working directory
WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# HuggingFace Spaces runs as non-root user 1000
RUN useradd -m -u 1000 user && chown -R user:user /app
USER user

# Create temp dir that the app can write to
RUN mkdir -p /tmp/cpptraj_sessions

# Expose port 7860 (required by HuggingFace Spaces)
EXPOSE 7860

ENV PORT=7860
ENV CPPTRAJ_PATH=/opt/conda/bin/cpptraj
ENV FLASK_SECRET_KEY=cpptrajgpt-hf-spaces-secret

CMD ["python", "server.py"]
