FROM continuumio/miniconda3:23.5.2-0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/lists/*

# Create isolated Python 3.11 environment with cpptraj bundled via ambertools
RUN conda create -n amber_env -c conda-forge python=3.11 ambertools -y && conda clean -afy

# Set working directory
WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies into base env
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 user && chown -R user:user /app
USER user

EXPOSE 8502

ENV PORT=8502
ENV CPPTRAJ_PATH=/opt/conda/envs/amber_env/bin/cpptraj
ENV FLASK_SECRET_KEY=cpptraj-ai-secret-change-me

CMD ["python", "server.py"]
