# Use Python 3.11 as the base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Create vectorstores directory and set permissions
RUN mkdir -p /app/vectorstores && \
    chmod 777 /app/vectorstores

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV VECTOR_STORE_PATH=/app/vectorstores

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "rag_tool.api:app", "--host", "0.0.0.0", "--port", "8000"] 