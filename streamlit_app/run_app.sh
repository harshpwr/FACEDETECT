#!/bin/bash

# Run Face Detection Streamlit App
# This script starts the Streamlit application

echo "======================================"
echo "AI Face Detection - Real vs Fake"
echo "======================================"
echo ""

# Check if model exists
if [ ! -f "models/face_detector_model.h5" ]; then
    echo "⚠️  Model not found!"
    echo "Please train the model first:"
    echo "  1. Run: jupyter notebook training_notebook.ipynb"
    echo "  2. Or run: python train.py"
    echo ""
    exit 1
fi

echo "✅ Model found: models/face_detector_model.h5"
echo "Starting Streamlit app..."
echo ""
echo "The app will open in your browser at: http://localhost:8501"
echo "Press Ctrl+C to stop the server"
echo ""

# Start Streamlit
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
