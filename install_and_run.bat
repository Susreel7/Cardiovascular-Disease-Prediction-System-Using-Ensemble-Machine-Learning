@echo off
echo ================================================================
echo Cardiovascular Disease Prediction System - Setup and Launch
echo ================================================================
echo.

echo [Step 1/4] Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/downloads/
    pause
    exit /b 1
)
echo.

echo [Step 2/4] Upgrading pip...
python -m pip install --upgrade pip --quiet
echo.

echo [Step 3/4] Installing required packages...
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install packages
    pause
    exit /b 1
)
echo.

echo [Step 4/4] Checking model files...
if not exist "ensemble_model.pkl" (
    echo WARNING: Model files not found!
    echo Do you want to train the model now? (This will take 5-10 minutes)
    echo Press Y to train, or N to skip and use existing model:
    set /p train_model="Train model? (Y/N): "
    if /i "%train_model%"=="Y" (
        echo Training model...
        python train_ensemble_model.py
    )
) else (
    echo Model files found - ready to launch!
)
echo.

echo [Launching Application...]
echo Application will open in your default browser at http://localhost:8501
echo.
echo To stop the application, press Ctrl+C in this window
echo.

python -m streamlit run streamlit_app_enhanced.py

pause

