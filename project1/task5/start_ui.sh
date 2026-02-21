#!/bin/bash

# Azerbaijani Spell Checker UI Startup Script

echo "🔤 Starting Azerbaijani Spell Checker UI..."
echo "=================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Start the application
echo "Starting Flask application..."
echo "The UI will be available at: http://localhost:5000"
echo "Press Ctrl+C to stop the server"
echo "=================================="

python spell_checker_ui.py