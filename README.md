# Cardiovascular Disease Prediction System Using Ensemble Machine Learning

## 📌 Overview
This project is an advanced **Cardiovascular Disease Prediction System** built with **Streamlit** and **Machine Learning**. It utilizes an **Ensemble Stacking Classifier** (combining Random Forest, XGBoost, etc.) to predict the likelihood of heart disease based on patient medical attributes.

The application provides a user-friendly interface for healthcare professionals to:
- Register and manage patients.
- Input clinical data (vitals, lab results).
- Get real-time risk assessments with detailed explanations.
- View comprehensive reports and visualizations.
- Track patient history over time.

## 🚀 Features
- **Ensemble Machine Learning Model**: High-accuracy prediction using a stacked ensemble of multiple classifiers.
- **Interactive Web Interface**: Built with Streamlit for easy usage.
- **Patient Management System**: Database integration (SQLite) to save and retrieve patient records.
- **Detailed Risk Reports**: Generates comprehensive reports explaining *why* a prediction was made, including contributing risk factors.
- **Visualizations**: Interactive charts for health metrics and risk analysis.
- **Clinical Interpretations**: Provides context for medical values (e.g., "Stage 2 Hypertension", "High Cholesterol").

## 📂 Project Structure
```
├── streamlit_app_enhanced.py    # Main Streamlit application
├── train_ensemble_model.py      # Script to train the ML model
├── patient_database.py          # Database management (SQLite)
├── FINAL_DATASET.csv            # Dataset used for training
├── requirements.txt             # Python dependencies
├── install_dependencies.bat     # Helper script for installation (Windows)
├── run_streamlit_app.bat        # Helper script to run the app (Windows)
└── models/                      # Directory where trained models are saved (.pkl files)
```

## 📊 Dataset
The project uses a comprehensive heart disease dataset (`FINAL_DATASET.csv`) containing the following clinical features:
- **Age**: Age in years.
- **Sex**: (1 = male; 0 = female).
- **Chest Pain Type**: (1 = typical angina; 2 = atypical angina; 3 = non-anginal pain; 4 = asymptomatic).
- **Resting Blood Pressure**: (in mm Hg).
- **Cholesterol**: Serum cholestoral in mg/dl.
- **Fasting Blood Sugar**: (1 = true; 0 = false).
- **Resting ECG**: (0 = normal; 1 = ST-T wave abnormality; 2 = LV hypertrophy).
- **Max Heart Rate**: Maximum heart rate achieved.
- **Exercise Angina**: (1 = yes; 0 = no).
- **Oldpeak**: ST depression induced by exercise relative to rest.
- **ST Slope**: The slope of the peak exercise ST segment.
- **Target**: Diagnosis of heart disease (1 = disease, 0 = no disease).

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Git

### Steps

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Susreel7/ensembleCardio.git
    cd ensembleCardio/Cardiovascular-Disease-Prediction-System-Using-Ensemble-Machine-Learning
    ```

2.  **Install Dependencies**
    You can use the provided batch file (Windows) or pip:
    ```bash
    pip install -r requirements.txt
    ```
    *Alternatively, run `install_dependencies.bat` on Windows.*

3.  **Train the Model**
    Before running the app, ensure the model is trained and saved:
    ```bash
    python train_ensemble_model.py
    ```
    *This will generate `ensemble_model.pkl` and other necessary artifacts.*

## 🏃‍♂️ Execution

To launch the application:

```bash
streamlit run streamlit_app_enhanced.py
```

*Alternatively, double-click `run_streamlit_app.bat` on Windows.*

The application will open in your default web browser at `http://localhost:8501`.

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

1.  Fork the project
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## 📜 License
This project is open source and available under the [MIT License](LICENSE).
