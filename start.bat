@echo off
REM LLM Evaluation Platform - Startup Script for Windows
REM This script sets up and starts the LLM Evaluation Platform

echo 🚀 Starting LLM Evaluation Platform...

REM Check if .env exists
if not exist .env (
    echo ⚠️  .env file not found. Creating from .env.example...
    copy .env.example .env
    echo 📝 Please edit .env file with your API keys before continuing.
    echo    Required: ANTHROPIC_API_KEY or OPENAI_API_KEY
    pause
)

REM Create necessary directories
echo 📁 Creating necessary directories...
if not exist data mkdir data
if not exist results mkdir results

REM Check if virtual environment exists
if not exist venv (
    echo 🔧 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔌 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install/update dependencies
echo 📦 Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

REM Check what to start
echo.
echo What would you like to start?
echo 1) Dashboard (Streamlit)
echo 2) API Server (FastAPI)
echo 3) Both Dashboard and API Server
echo 4) Run Evaluation (CLI)
set /p choice="Enter choice (1-4): "

if "%choice%"=="1" (
    echo 🎨 Starting Dashboard...
    streamlit run dashboard.py
) else if "%choice%"=="2" (
    echo 🔌 Starting API Server...
    python api_server.py
) else if "%choice%"=="3" (
    echo 🎨 Starting Dashboard in background...
    start "Dashboard" streamlit run dashboard.py
    
    echo 🔌 Starting API Server in background...
    start "API Server" python api_server.py
    
    echo ✅ Dashboard running at http://localhost:8501
    echo ✅ API Server running at http://localhost:8000
    echo 📚 API Documentation: http://localhost:8000/docs
    echo.
    echo Press any key to close this window (services will continue running)
    pause
) else if "%choice%"=="4" (
    echo 🔬 Running Evaluation...
    python run_evaluation.py
) else (
    echo ❌ Invalid choice. Exiting.
    exit /b 1
)
