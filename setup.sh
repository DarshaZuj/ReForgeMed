#!/bin/bash

# Create necessary directories
mkdir -p .cache/primekg

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:${PWD}"

# Download any required data files here if needed
# Example:
# wget -P .cache/primekg/ https://dataverse.harvard.edu/api/access/datafile/6043893 -O .cache/primekg/primekg.parquet

# Install Python dependencies
pip install -r requirements.txt
