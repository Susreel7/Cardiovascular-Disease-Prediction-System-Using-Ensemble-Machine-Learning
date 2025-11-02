@echo off
echo ============================================
echo Installing Dependencies for Heart Disease Prediction
echo ============================================
echo.

echo Checking Python version...
python --version
echo.

echo Installing required packages...
python -m pip install --upgrade pip
python -m pip install pandas numpy scikit-learn xgboost joblib streamlit scipy

echo.
echo ============================================
echo Installation Complete!
echo ============================================
echo.
echo Next steps:
echo 1. Run: python train_ensemble_model.py
echo 2. Run: streamlit run streamlit_app.py
echo.
pause
