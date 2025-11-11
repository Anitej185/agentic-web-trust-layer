#!/bin/bash

# Financial Intelligence System Startup Script

echo "🚀 Starting Financial Intelligence System..."
echo "================================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/upgrade dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check for .env file
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found!"
    echo "📝 Creating .env from template..."
    cp .env.example .env
    echo "Please edit .env file and add your API keys"
    echo "At minimum, you need either OPENAI_API_KEY or GROQ_API_KEY"
    read -p "Press enter to continue after adding API keys..."
fi

# Start backend in background
echo "🔧 Starting backend server..."
python backend_enhanced.py &
BACKEND_PID=$!

# Wait for backend to start
echo "⏳ Waiting for backend to initialize..."
sleep 5

# Check if backend is running
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Backend is running!"
else
    echo "❌ Backend failed to start. Check logs."
    exit 1
fi

# Start frontend
echo "🎨 Starting frontend..."
streamlit run app_enhanced.py

# Cleanup on exit
trap "kill $BACKEND_PID 2>/dev/null" EXIT
