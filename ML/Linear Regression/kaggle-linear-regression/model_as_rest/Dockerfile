# Stage 1: Build stage
FROM python:3.11-slim AS builder

# Set the working directory
WORKDIR /app

# Install build dependencies (e.g., gcc for compiling packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file
COPY requirements.txt .

# Install dependencies in a virtual environment
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# Stage 2: Final stage (runtime)
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Ensure venv is enabled when running Python
ENV PATH="/opt/venv/bin:$PATH"

# Copy the application code from the current directory to the container
COPY . .

# Expose the port that Uvicorn will run on
EXPOSE 8000

# Command to run the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
