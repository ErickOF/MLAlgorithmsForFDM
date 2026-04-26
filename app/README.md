# MEX/FDM Quality Analyzer

Interactive web application for MEX/FDM 3D printing quality analysis. Runs regression and classification models, hyperparameter sweeps, and optimization on your process data.

---

## Requirements

- **Python 3.10+**
- A terminal (bash/zsh on Linux/macOS, Command Prompt or PowerShell on Windows)

---

## Setup

### Linux

```bash
cd app/
chmod +x setup.sh
./setup.sh
```

### macOS

```bash
cd app/
chmod +x setup_macos.sh
./setup_macos.sh
```

Requires Python 3.10+ installed via [Homebrew](https://brew.sh) (`brew install python`) or the official installer.

### Windows — Command Prompt

```bat
cd app\
setup.bat
```

### Windows — PowerShell

```powershell
cd app\
# If scripts are blocked by execution policy, run this once:
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

.\setup.ps1
```

All scripts:
1. Create a `.venv/` virtual environment
2. Activate it and upgrade `pip`
3. Install all packages listed in `requirements.txt`

---

## Running the app

After setup, activate the environment and start the server:

### Linux / macOS

```bash
source .venv/bin/activate
python app.py
```

### Windows — Command Prompt

```bat
.venv\Scripts\activate.bat
python app.py
```

### Windows — PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
python app.py
```

Then open your browser at [http://127.0.0.1:5000](http://127.0.0.1:5000).

### Optional flags

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | `5000` | Port to listen on |
| `--host` | `127.0.0.1` | Host address (`0.0.0.0` to expose on the network) |

Example:

```bash
python app.py --port 8080 --host 0.0.0.0
```

---

## Data

Place your Excel data file in `data/` and update `DATA_PATH` / `SHEET_NAME` at the top of `app.py` if the filename or sheet differs from the default.

---

## Project structure

```
app/
├── app.py               # Flask web application
├── requirements.txt     # Python dependencies
├── setup.sh             # Linux setup script
├── setup_macos.sh       # macOS setup script
├── setup.bat            # Windows Command Prompt setup script
├── setup.ps1            # Windows PowerShell setup script
├── data/                # Input data files
└── src/
    ├── run_quality_analysis.py   # Model training & evaluation engine
    ├── optimize_quality.py       # Optimization routines
    ├── run_from_config.py        # CLI: run analysis from exported JSON config
    └── run_optimization.py       # CLI: run optimization from command line
```
