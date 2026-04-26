#!/usr/bin/env bash
# setup.sh – Create the virtual environment and install dependencies
set -euo pipefail

VENV_DIR=".venv"

echo ">>> Creating virtual environment in ${VENV_DIR}/"
python3 -m venv "${VENV_DIR}"

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
