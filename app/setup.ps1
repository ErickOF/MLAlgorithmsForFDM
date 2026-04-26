# setup.ps1 – Create the virtual environment and install dependencies
# Run from a PowerShell terminal inside the app/ directory.
# If execution policy blocks the script, run first:
#   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

$ErrorActionPreference = "Stop"

$VenvDir = ".venv"

Write-Host ">>> Creating virtual environment in $VenvDir/"
python -m venv $VenvDir

Write-Host ">>> Activating virtual environment"
& "$VenvDir\Scripts\Activate.ps1"

Write-Host ">>> Upgrading pip"
python -m pip install --upgrade pip

Write-Host ">>> Installing dependencies from requirements.txt"
pip install -r requirements.txt

Write-Host ""
Write-Host "Setup complete."
Write-Host "To activate the environment in future sessions run:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Then start the app with:"
Write-Host "  python app.py"
