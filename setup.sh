#!/usr/bin/env bash
# Run this in Git Bash or a Unix shell
#   ./setup.sh
set -e  # exit on first error

# === Always run from the directory where this script lives ===
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

API_DIR="API"
VENV_DIR="$API_DIR/.venv"

echo "=== Creating virtual environment in: $VENV_DIR ==="
python -m venv "$VENV_DIR"

echo "=== Activating virtual environment ==="
if [[ -d "$VENV_DIR/Scripts" ]]; then
    # Windows (Git Bash uses Windows Python)
    # shellcheck disable=SC1090
    source "$VENV_DIR/Scripts/activate"
else
    # macOS / Linux
    # shellcheck disable=SC1090
    source "$VENV_DIR/bin/activate"
fi

echo "=== Upgrading pip & installing requirements ==="
python -m pip install --upgrade pip
pip install -r "$API_DIR/requirements.txt"

echo "=== Changing into API directory ==="
cd "$API_DIR"

echo "=== Starting Flask server: python app.py ==="
python app.py

# Script will stay here until you Ctrl+C the server
