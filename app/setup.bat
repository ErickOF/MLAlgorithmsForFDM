@echo off
REM setup.bat – Create the virtual environment and install dependencies
REM Run from the app\ directory in Command Prompt.

SET VENV_DIR=.venv

echo ^>^>^> Creating virtual environment in %VENV_DIR%\
python -m venv %VENV_DIR%
IF ERRORLEVEL 1 (
    echo ERROR: Failed to create virtual environment.
    echo Make sure Python 3.10+ is installed and on your PATH.
    exit /b 1
)

echo ^>^>^> Activating virtual environment
CALL %VENV_DIR%\Scripts\activate.bat
IF ERRORLEVEL 1 (
    echo ERROR: Failed to activate virtual environment.
    exit /b 1
)

echo ^>^>^> Upgrading pip
python -m pip install --upgrade pip

echo ^>^>^> Installing dependencies from requirements.txt
pip install -r requirements.txt
IF ERRORLEVEL 1 (
    echo ERROR: Dependency installation failed.
    exit /b 1
)

echo.
echo Setup complete.
echo To activate the environment in future sessions run:
echo   %VENV_DIR%\Scripts\activate.bat
echo.
echo Then start the app with:
echo   python app.py
