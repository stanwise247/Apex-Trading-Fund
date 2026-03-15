#!/bin/bash

echo ""
echo " ============================================"
echo "  APEX Trading Engine — Phase 1 Setup"
echo " ============================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo " ERROR: Python 3 not found."
    echo " Install via: brew install python3 (Mac)"
    echo " or: sudo apt install python3 python3-pip (Linux)"
    exit 1
fi

echo " [1/3] Python found: $(python3 --version)"
echo " [2/3] Installing dependencies..."
pip3 install -r requirements.txt --quiet

if [ $? -ne 0 ]; then
    echo " ERROR: Failed to install dependencies."
    echo " Try: pip3 install flask flask-cors requests"
    exit 1
fi

echo " [3/3] Starting APEX Engine..."
echo ""
echo " ============================================"
echo "  Server running at: http://localhost:5000"
echo "  Open apex_dashboard.html in your browser"
echo " ============================================"
echo ""
echo " Add your API keys in the dashboard Setup menu"
echo " or edit config.json directly."
echo ""
echo " Press Ctrl+C to stop."
echo ""

python3 server.py
