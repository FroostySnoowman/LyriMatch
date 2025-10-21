#I havent actually tested this so let me know if this sets up the VENV correctly
set -e 

VENV_DIR="venv-lyrics"

echo "=== Creating virtual environment: $VENV_DIR ==="
python -m venv "$VENV_DIR"

# Spin up virtual environment
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    source "$VENV_DIR/Scripts/activate"
else
    source "$VENV_DIR/bin/activate"
fi

python -m pip install --upgrade pip
pip install -r requirements.txt


