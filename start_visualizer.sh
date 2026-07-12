#!/bin/bash

# start_visualizer.sh
# Unified launcher for the DRS visualizer system (React frontend + Python dev server)

set -e

# Clear screen for a clean console layout
clear

echo -e "\033[1;36m==================================================\033[0m"
echo -e "\033[1;36m              DRS Visualizer System               \033[0m"
echo -e "\033[1;36m==================================================\033[0m"

# Kill any existing process on port 8000 to ensure clean startup
kill $(lsof -ti :8000) 2>/dev/null || true

echo -e "\033[0;32m[+] Starting Python Local Dev Server...\033[0m"

# Start the Python dev server in the background
.venv/bin/python drs-canvas/drs_dev_server.py &
PYTHON_PID=$!

# Ensure the python server is terminated when Ctrl+C is pressed
cleanup() {
  echo -e "\n\033[0;33m[-] Shutting down DRS Visualizer system...\033[0m"
  kill $PYTHON_PID 2>/dev/null || true
  echo -e "\033[0;32m[+] Cleanup complete. Goodbye!\033[0m"
}
trap cleanup EXIT

# Wait a brief moment for python server to bind port 8000
sleep 1.5

echo -e "\033[0;32m[+] Starting Vite React Frontend...\033[0m"
echo -e "\033[0;35m[i] Note: Open http://localhost:5173 in your browser to view the editor.\033[0m"
echo -e "\033[1;36m--------------------------------------------------\033[0m"

# Start the Vite React app dev server
npm --prefix drs-canvas run dev
