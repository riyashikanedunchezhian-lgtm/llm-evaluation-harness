#!/bin/bash

# LLM Evaluation Platform - Startup Script
# This script sets up and starts the LLM Evaluation Platform

set -e

echo "🚀 Starting LLM Evaluation Platform..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Creating from .env.example..."
    cp .env.example .env
    echo "📝 Please edit .env file with your API keys before continuing."
    echo "   Required: ANTHROPIC_API_KEY or OPENAI_API_KEY"
    read -p "Press Enter after editing .env file..."
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p data results

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check what to start
echo ""
echo "What would you like to start?"
echo "1) Dashboard (Streamlit)"
echo "2) API Server (FastAPI)"
echo "3) Both Dashboard and API Server"
echo "4) Run Evaluation (CLI)"
read -p "Enter choice (1-4): " choice

case $choice in
    1)
        echo "🎨 Starting Dashboard..."
        streamlit run dashboard.py
        ;;
    2)
        echo "🔌 Starting API Server..."
        python api_server.py
        ;;
    3)
        echo "🎨 Starting Dashboard in background..."
        streamlit run dashboard.py &
        DASHBOARD_PID=$!
        
        echo "🔌 Starting API Server in background..."
        python api_server.py &
        API_PID=$!
        
        echo "✅ Dashboard running at http://localhost:8501 (PID: $DASHBOARD_PID)"
        echo "✅ API Server running at http://localhost:8000 (PID: $API_PID)"
        echo "📚 API Documentation: http://localhost:8000/docs"
        echo ""
        echo "Press Ctrl+C to stop both services"
        
        # Wait for both processes
        wait $DASHBOARD_PID $API_PID
        ;;
    4)
        echo "🔬 Running Evaluation..."
        python run_evaluation.py
        ;;
    *)
        echo "❌ Invalid choice. Exiting."
        exit 1
        ;;
esac
