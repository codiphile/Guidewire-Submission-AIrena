FROM python:3.10-slim

WORKDIR /app

# Install required packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p src/data/predictions src/data/archive \
    src/models/saved \
    src/metrics/predictions \
    src/plots/archive

# Generate sample data
RUN python main.py generate-data --num-records 10000 --failure-rate 0.05 --output-path src/data/generated_metrics.csv --skip-training --skip-testing

# Train the model
RUN python main.py train-svm --data-path src/data/generated_metrics.csv --grid-search --visualize

# Expose port for API
EXPOSE 5000

# Set the entry point
ENTRYPOINT ["python", "-m", "src.api"]