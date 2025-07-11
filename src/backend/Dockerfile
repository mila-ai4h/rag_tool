# Use an official Python runtime as a base image.
FROM python:3.11-slim

# Set environment variables.
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Set the working directory.
WORKDIR /app

# Copy the requirements file into the container.
COPY src/backend/requirements.txt /app/

# Install system dependencies needed for building some Python packages
RUN apt-get update && apt-get install -y gcc g++ make && rm -rf /var/lib/apt/lists/*

# uv pip
RUN pip install uv
RUN pip install --upgrade pip setuptools wheel
RUN uv pip install -r requirements.txt --system

# Copy the entire backend folder (including your API code, celery tasks, etc.) into the container.
COPY src/backend/ /app/

# Expose the port Cloud Run (or your local environment) expects.
EXPOSE 8080

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
