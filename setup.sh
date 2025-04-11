#!/bin/bash

# Create a virtual environment using uv
echo "Creating virtual environment..."
uv venv

# Activate the virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

echo "Installing dependencies..."
uv pip install flask

# Initialize uploads directory if it doesn't exist
mkdir -p uploads

# Make sure instance directory exists
mkdir -p instance

# Run the Flask application
echo "Starting Flask server..."
python3 app.py