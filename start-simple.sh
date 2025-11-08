#!/bin/bash

# ARES React Dashboard - Simple Development Startup
# Works without Docker - just starts backend and frontend

set -e  # Exit on error

echo "🤖 ARES React Dashboard - Starting Development Servers"
echo "======================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if ARES data exists
echo "📁 Checking ARES data..."
if [ -f "data/robot_data.db" ]; then
    echo -e "${GREEN}✅ Database file found${NC}"
else
    echo -e "${RED}❌ Database not found at data/robot_data.db${NC}"
    echo -e "${YELLOW}Please run 'python main.py' first to ingest data${NC}"
    exit 1
fi

echo ""

# Check if backend dependencies are installed
echo "🐍 Checking backend dependencies..."
cd backend
if python -c "import fastapi" 2>/dev/null; then
    echo -e "${GREEN}✅ Backend dependencies installed${NC}"
else
    echo -e "${YELLOW}⚠️  Installing backend dependencies...${NC}"
    pip install -r requirements.txt
    echo -e "${GREEN}✅ Backend dependencies installed${NC}"
fi
cd ..

echo ""

# Check if frontend dependencies are installed
echo "📦 Checking frontend dependencies..."
cd frontend
if [ -d "node_modules" ]; then
    echo -e "${GREEN}✅ Frontend dependencies installed${NC}"
else
    echo -e "${YELLOW}⚠️  Installing frontend dependencies...${NC}"
    npm install
    echo -e "${GREEN}✅ Frontend dependencies installed${NC}"
fi
cd ..

echo ""
echo "🚀 Starting servers..."
echo ""
echo -e "${GREEN}Backend will run on: http://localhost:8000${NC}"
echo -e "${GREEN}Frontend will run on: http://localhost:5173${NC}"
echo ""
echo -e "${YELLOW}Note: MongoDB is optional. The app will work without annotations.${NC}"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""
sleep 2

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping servers..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit 0
}

trap cleanup INT TERM

# Start backend in background
echo "Starting backend..."
cd backend
python main.py &
BACKEND_PID=$!
cd ..

# Wait for backend to start
echo "Waiting for backend to initialize..."
sleep 5

# Start frontend in background
echo "Starting frontend..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo -e "${GREEN}✅ Servers started!${NC}"
echo ""
echo "Open http://localhost:5173 in your browser"
echo ""

# Wait for both processes
wait
