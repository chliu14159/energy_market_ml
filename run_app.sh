#!/bin/bash

# Retail Analytics Platform - Streamlit App Launcher
echo "🚀 Starting Retail Analytics Platform..."

# Check if we're in the right directory
if [ ! -f "app/main.py" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    echo "Expected structure: app/main.py should exist"
    exit 1
fi

# Check if dev_ml virtual environment exists
if [ ! -f "../../dev_ml/bin/activate" ]; then
    echo "❌ Error: dev_ml virtual environment not found"
    echo "Expected location: ../../dev_ml/bin/activate"
    echo "Please create the virtual environment or run manually with your Python environment"
    exit 1
fi

# Activate the virtual environment
echo "🔧 Activating dev_ml virtual environment..."
source ../../dev_ml/bin/activate

# Check if data exists
if [ ! -f "output/unified_analytics.parquet" ]; then
    echo "⚠️  Warning: Analytics data not found at output/unified_analytics.parquet"
    echo "Please run the data preparation script first:"
    echo "cd scripts && python data_preparation.py"
    echo ""
    read -p "Continue anyway? (y/N): " continue_choice
    if [[ ! $continue_choice =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Launch Streamlit
echo "🌐 Opening Streamlit app with dev_ml environment..."
echo "📍 URL: http://localhost:8501"
echo "⏹️  To stop: Press Ctrl+C"
echo ""

# Install missing packages if needed
echo "📦 Checking for required packages..."
pip install statsmodels --quiet

streamlit run app/main.py --server.port 8501 --server.address localhost
