#!/usr/bin/env bash
# setup_macos.sh – Create the virtual environment and install dependencies
# Designed for macOS (Homebrew Python or system Python 3).
set -euo pipefail

VENV_DIR=".venv"

# Prefer python3 from Homebrew if available, otherwise fall back to system python3
if command -v python3 &>/dev/null; then
    PYTHON=python3
else
    echo "ERROR: python3 not found. Install it via Homebrew: brew install python" >&2
    exit 1
fi

echo ">>> Using $($PYTHON --version)"

echo ">>> Creating virtual environment in ${VENV_DIR}/"
$PYTHON -m venv "${VENV_DIR}"

echo ">>> Activating virtual environment"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo ">>> Upgrading pip"
pip install --upgrade pip

echo ">>> Installing dependencies from requirements.txt"
pip install -r requirements.txt

echo ""
echo "Setup complete."
echo "To activate the environment in future sessions run:"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
echo "Then start the app with:"
echo "  python app.py"
