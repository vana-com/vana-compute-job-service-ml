FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Copy poetry configuration files and README first
COPY app/pyproject.toml app/poetry.lock* /app/
COPY README.md /app/

# Configure poetry to not use a virtual environment
RUN poetry config virtualenvs.create false

# Copy application code
COPY app/ /app/

# Install dependencies
RUN poetry install --without dev --no-interaction --no-ansi

# Create directories for output, and working data
RUN mkdir -p /mnt/output /mnt/working

# Set environment variables
ENV OUTPUT_PATH=/mnt/output
ENV WORKING_PATH=/mnt/working
ENV PORT=8000

# Expose the port
EXPOSE 8000

# Run the application
CMD ["python", "-m", "main"]