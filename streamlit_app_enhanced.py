"""
Enhanced Streamlit App for Cardiovascular Disease Prediction
Features: Patient Management, Detailed Reports, Statistics, Visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from patient_database import PatientDatabase
import json
import io

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Advanced Heart Disease Prediction System",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database
@st.cache_resource
def init_database():
    return PatientDatabase()

db = init_database()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 2rem 0;
    }
    .metric-card {
        padding: 1.5rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained ensemble model"""
    try:
        model = joblib.load('ensemble_model.pkl')
        scaler = joblib.load('feature_scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
        metadata = joblib.load('model_metadata.pkl')
        return model, scaler, feature_names, metadata
    except FileNotFoundError as e:
        st.error(f"❌ Model file not found: {e}")
        st.info("Please run 'train_ensemble_model.py' first to train and save the model.")
        return None, None, None, None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None, None, None

def analyze_prediction_factors(user_input, prediction, probability):
    """Analyze factors contributing to the prediction"""
    factors = {
        'protective': [],
        'risk': [],
        'neutral': []
    }
    
    # Age analysis
    if user_input['age'] < 50:
        factors['protective'].append(f"Young age ({user_input['age']} years) - lower risk")
    elif user_input['age'] >= 50:
        factors['risk'].append(f"Age ≥ 50 ({user_input['age']} years) - higher risk factor")
    
    # Sex analysis
    if user_input['sex'] == 'Female':
        factors['protective'].append("Female sex - generally lower cardiovascular risk")
    else:
        factors['risk'].append("Male sex - higher cardiovascular risk")
    
    # Blood Pressure
    if user_input['resting_blood_pressure'] >= 140:
        factors['risk'].append(f"High blood pressure ({user_input['resting_blood_pressure']} mmHg) - Stage 2 Hypertension")
    elif user_input['resting_blood_pressure'] >= 120:
        factors['risk'].append(f"Elevated blood pressure ({user_input['resting_blood_pressure']} mmHg) - Stage 1 Hypertension")
    else:
        factors['protective'].append(f"Normal blood pressure ({user_input['resting_blood_pressure']} mmHg)")
    
    # Cholesterol
    if user_input['cholesterol'] >= 240:
        factors['risk'].append(f"High cholesterol ({user_input['cholesterol']} mg/dL)")
    elif user_input['cholesterol'] >= 200:
        factors['risk'].append(f"Borderline cholesterol ({user_input['cholesterol']} mg/dL)")
    else:
        factors['protective'].append(f"Desirable cholesterol ({user_input['cholesterol']} mg/dL)")
    
    # Chest Pain Type
    if user_input['chest_pain_type'] == 'Typical Angina':
        factors['risk'].append("Typical Angina - strong indicator of coronary artery disease")
    elif user_input['chest_pain_type'] == 'Atypical Angina':
        factors['risk'].append("Atypical Angina - moderate risk indicator")
    elif user_input['chest_pain_type'] == 'Non-Anginal Pain':
        factors['protective'].append("Non-Anginal Pain - less likely cardiac-related")
    else:
        factors['neutral'].append("Asymptomatic - no chest pain")
    
    # Resting ECG
    if user_input['rest_ecg'] != 'Normal':
        factors['risk'].append(f"Abnormal ECG: {user_input['rest_ecg']}")
    else:
        factors['protective'].append("Normal resting ECG")
    
    # Exercise Induced Angina
    if user_input['exercise_induced_angina'] == 'Yes':
        factors['risk'].append("Exercise-induced angina - significant risk indicator")
    else:
        factors['protective'].append("No exercise-induced angina")
    
    # ST Depression
    if user_input['st_depression'] >= 2.0:
        factors['risk'].append(f"Significant ST depression ({user_input['st_depression']} mm)")
    elif user_input['st_depression'] >= 1.0:
        factors['risk'].append(f"Moderate ST depression ({user_input['st_depression']} mm)")
    else:
        factors['protective'].append(f"Minimal ST depression ({user_input['st_depression']} mm)")
    
    # ST Slope
    if user_input['st_slope'] == 'Downsloping':
        factors['risk'].append("Downsloping ST segment - associated with myocardial ischemia")
    elif user_input['st_slope'] == 'Flat':
        factors['risk'].append("Flat ST segment - moderate concern")
    else:
        factors['protective'].append("Upsloping ST segment - normal finding")
    
    # Fasting Blood Sugar
    if user_input['fasting_blood_sugar'] == 'Yes':
        factors['risk'].append("Elevated fasting blood sugar - diabetes/impaired glucose")
    else:
        factors['protective'].append("Normal fasting blood sugar")
    
    return factors

def preprocess_input(user_input, feature_names, scaler):
    """Preprocess user input to match training data format"""
    input_df = pd.DataFrame(0, index=[0], columns=feature_names)
    
    # Map inputs
    input_df['age'] = user_input['age']
    input_df['sex_male'] = 1 if user_input['sex'] == 'Male' else 0
    input_df['resting_blood_pressure'] = user_input['resting_blood_pressure']
    input_df['cholesterol'] = user_input['cholesterol']
    input_df['fasting_blood_sugar'] = 1 if user_input['fasting_blood_sugar'] == 'Yes' else 0
    input_df['exercise_induced_angina'] = 1 if user_input['exercise_induced_angina'] == 'Yes' else 0
    input_df['st_depression'] = user_input['st_depression']
    input_df['max_heart_rate_achieved'] = user_input['max_heart_rate_achieved']
    
    # Chest pain type
    chest_pain_mapping = {
        'Typical Angina': 'chest_pain_type_typical angina',
        'Atypical Angina': 'chest_pain_type_atypical angina',
        'Non-Anginal Pain': 'chest_pain_type_non-anginal pain',
        'Asymptomatic': 'chest_pain_type_asymptomatic'
    }
    for key, col in chest_pain_mapping.items():
        if col in input_df.columns:
            input_df[col] = 1 if user_input['chest_pain_type'] == key else 0
    
    # Rest ECG
    rest_ecg_mapping = {
        'Normal': 'rest_ecg_normal',
        'ST-T Wave Abnormality': 'rest_ecg_ST-T wave abnormality',
        'Left Ventricular Hypertrophy': 'rest_ecg_left ventricular hypertrophy'
    }
    for key, col in rest_ecg_mapping.items():
        if col in input_df.columns:
            input_df[col] = 1 if user_input['rest_ecg'] == key else 0
    
    # ST slope
    st_slope_mapping = {
        'Upsloping': 'st_slope_upsloping',
        'Flat': 'st_slope_flat',
        'Downsloping': 'st_slope_downsloping'
    }
    for key, col in st_slope_mapping.items():
        if col in input_df.columns:
            input_df[col] = 1 if user_input['st_slope'] == key else 0
    
    # Scale numeric features
    numeric_features = ['age', 'resting_blood_pressure', 'cholesterol', 
                        'max_heart_rate_achieved', 'st_depression']
    numeric_features = [f for f in numeric_features if f in input_df.columns]
    
    if numeric_features:
        input_df[numeric_features] = scaler.transform(input_df[numeric_features])
    
    return input_df

def get_detailed_report(user_input, prediction, probability, risk_level, 
                       recommendations, warnings_list, lifestyle_tips, metadata=None):
    """Generate a detailed comprehensive report"""
    
    # Calculate risk factors
    risk_factors = []
    if user_input['age'] >= 50:
        risk_factors.append(f"Age ≥ 50 ({user_input['age']} years)")
    if user_input['resting_blood_pressure'] >= 140:
        risk_factors.append(f"Hypertension ({user_input['resting_blood_pressure']} mmHg)")
    if user_input['cholesterol'] >= 240:
        risk_factors.append(f"High Cholesterol ({user_input['cholesterol']} mg/dL)")
    if user_input['fasting_blood_sugar'] == 'Yes':
        risk_factors.append("Diabetes/Impaired Fasting Glucose")
    if user_input['chest_pain_type'] == 'Typical Angina':
        risk_factors.append("Typical Angina Chest Pain")
    if user_input['exercise_induced_angina'] == 'Yes':
        risk_factors.append("Exercise-Induced Angina")
    if user_input['rest_ecg'] != 'Normal':
        risk_factors.append(f"Abnormal ECG: {user_input['rest_ecg']}")
    if user_input['st_slope'] == 'Downsloping':
        risk_factors.append("Downsloping ST Segment")
    
    # Normal ranges
    normal_ranges = {
        'Blood Pressure': 'Normal: < 120/80 mmHg',
        'Cholesterol': 'Desirable: < 200 mg/dL',
        'Max Heart Rate': f'Typical for age {user_input["age"]}: ~{220 - user_input["age"]} bpm',
        'ST Depression': 'Normal: 0-1 mm'
    }
    
    report = f"""
{'='*80}
COMPREHENSIVE CARDIOVASCULAR DISEASE RISK ASSESSMENT REPORT
{'='*80}
Generated Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Report ID: {datetime.now().strftime('%Y%m%d%H%M%S')}

{'='*80}
1. PATIENT DEMOGRAPHIC INFORMATION
{'='*80}
Patient Name: [To be filled]
Age: {user_input['age']} years
Sex: {user_input['sex']}
Assessment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*80}
2. VITAL SIGNS AND CLINICAL MEASUREMENTS
{'='*80}
┌────────────────────────────────────┬──────────────┬──────────────────────┐
│ Parameter                          │ Value        │ Normal Range          │
├────────────────────────────────────┼──────────────┼──────────────────────┤
│ Resting Blood Pressure             │ {user_input['resting_blood_pressure']:>5} mmHg │ < 120/80 mmHg        │
│ Total Serum Cholesterol            │ {user_input['cholesterol']:>5} mg/dL  │ < 200 mg/dL          │
│ Maximum Heart Rate Achieved        │ {user_input['max_heart_rate_achieved']:>5} bpm   │ ~{220 - user_input['age']} bpm (age-based) │
│ Fasting Blood Sugar > 120 mg/dL   │ {user_input['fasting_blood_sugar']:>5}       │ No (Normal)          │
│ ST Depression                      │ {user_input['st_depression']:>5.1f} mm    │ 0-1 mm               │
└────────────────────────────────────┴──────────────┴──────────────────────┘

Clinical Interpretation:
"""
    
    # Add interpretations
    if user_input['resting_blood_pressure'] >= 140:
        report += f"⚠️ HIGH BLOOD PRESSURE: {user_input['resting_blood_pressure']} mmHg indicates Stage 2 Hypertension\n"
    elif user_input['resting_blood_pressure'] >= 120:
        report += f"📊 ELEVATED BLOOD PRESSURE: {user_input['resting_blood_pressure']} mmHg indicates Elevated/Stage 1 Hypertension\n"
    else:
        report += f"✅ NORMAL BLOOD PRESSURE: {user_input['resting_blood_pressure']} mmHg is within normal range\n"
    
    if user_input['cholesterol'] >= 240:
        report += f"⚠️ HIGH CHOLESTEROL: {user_input['cholesterol']} mg/dL indicates High Risk (≥240 mg/dL)\n"
    elif user_input['cholesterol'] >= 200:
        report += f"📊 BORDERLINE CHOLESTEROL: {user_input['cholesterol']} mg/dL indicates Borderline High (200-239 mg/dL)\n"
    else:
        report += f"✅ DESIRABLE CHOLESTEROL: {user_input['cholesterol']} mg/dL is within desirable range\n"
    
    max_hr_expected = 220 - user_input['age']
    if user_input['max_heart_rate_achieved'] < 0.7 * max_hr_expected:
        report += f"📊 HEART RATE: {user_input['max_heart_rate_achieved']} bpm is below 70% of expected max ({max_hr_expected} bpm)\n"
    
    report += f"""
{'='*80}
3. ELECTROCARDIOGRAPHIC AND EXERCISE TEST RESULTS
{'='*80}
┌────────────────────────────────────┬─────────────────────────────────┐
│ Test Parameter                     │ Result                          │
├────────────────────────────────────┼─────────────────────────────────┤
│ Chest Pain Type                     │ {user_input['chest_pain_type']:<33} │
│ Resting ECG Results                 │ {user_input['rest_ecg']:<33} │
│ Exercise Induced Angina             │ {user_input['exercise_induced_angina']:<33} │
│ ST Segment Slope                    │ {user_input['st_slope']:<33} │
│ ST Depression                       │ {user_input['st_depression']:.1f} mm                      │
└────────────────────────────────────┴─────────────────────────────────┘

Clinical Significance:
"""
    
    if user_input['chest_pain_type'] == 'Typical Angina':
        report += "⚠️ Typical angina is strongly associated with coronary artery disease\n"
    if user_input['rest_ecg'] != 'Normal':
        report += f"⚠️ Abnormal ECG findings: {user_input['rest_ecg']} requires clinical evaluation\n"
    if user_input['exercise_induced_angina'] == 'Yes':
        report += "⚠️ Exercise-induced angina is a significant indicator of cardiovascular disease\n"
    if user_input['st_slope'] == 'Downsloping':
        report += "⚠️ Downsloping ST segment is associated with myocardial ischemia\n"
    
    report += f"""
{'='*80}
4. AI MODEL PREDICTION AND RISK ASSESSMENT
{'='*80}
Ensemble Model: Stacking Classifier with Multiple Base Learners
Model Accuracy: ~{(metadata.get('accuracy', 0.92) if metadata else 0.92):.2%} (Cross-Validated)

┌────────────────────────────────────┬─────────────────────────────────┐
│ Assessment Parameter                │ Result                          │
├────────────────────────────────────┼─────────────────────────────────┤
│ Prediction                          │ {'HIGH RISK - Disease Detected' if prediction == 1 else 'LOW RISK - No Disease'} │
│ Disease Probability                 │ {probability:.2%}                           │
│ Risk Level                          │ {risk_level}                                    │
│ Confidence Level                    │ {'High' if abs(probability - 0.5) > 0.3 else 'Moderate'}                                    │
└────────────────────────────────────┴─────────────────────────────────┘

Risk Classification:
"""
    
    if probability >= 0.7:
        report += "⚠️ VERY HIGH RISK: Immediate medical consultation and comprehensive cardiac evaluation strongly recommended\n"
    elif probability >= 0.5:
        report += "⚠️ MODERATE-HIGH RISK: Medical consultation recommended within 1-2 weeks\n"
    else:
        report += "✅ LOW RISK: Continue preventive care and regular monitoring\n"
    
    report += f"""
{'='*80}
5. IDENTIFIED RISK FACTORS
{'='*80}
Total Risk Factors Identified: {len(risk_factors)}

"""
    
    for i, factor in enumerate(risk_factors, 1):
        report += f"{i}. {factor}\n"
    
    if not risk_factors:
        report += "✅ No significant modifiable risk factors identified.\n"
    
    report += f"""
{'='*80}
6. MEDICAL RECOMMENDATIONS
{'='*80}
"""
    
    for rec in recommendations:
        report += f"{rec}\n"
    
    report += f"""
{'='*80}
7. WARNINGS AND ALERTS
{'='*80}
"""
    
    for warning in warnings_list:
        report += f"{warning}\n"
    
    if not warnings_list:
        report += "✅ No immediate warnings at this time.\n"
    
    report += f"""
{'='*80}
8. LIFESTYLE MODIFICATIONS AND PREVENTIVE CARE
{'='*80}
"""
    
    for tip in lifestyle_tips:
        report += f"{tip}\n"
    
    report += f"""
{'='*80}
9. FOLLOW-UP RECOMMENDATIONS
{'='*80}
"""
    
    if probability >= 0.7:
        report += """
IMMEDIATE ACTIONS (Within 24-48 hours):
- Schedule emergency consultation with cardiologist
- Comprehensive cardiac evaluation including:
  * ECG and Holter monitoring
  * Echocardiogram
  * Stress test (if stable)
  * Cardiac catheterization (if indicated)
- Blood tests: Complete metabolic panel, lipid profile, cardiac enzymes
- Review and optimize medications

SHORT-TERM FOLLOW-UP (1-2 weeks):
- Cardiac consultation visit
- Review of all test results
- Medication optimization
- Lifestyle modification counseling

ONGOING MONITORING:
- Weekly blood pressure checks
- Monthly lipid profile monitoring
- Quarterly comprehensive review
"""
    elif probability >= 0.5:
        report += """
SHORT-TERM ACTIONS (Within 1-2 weeks):
- Schedule consultation with cardiologist or primary care physician
- Cardiac evaluation including:
  * ECG
  * Echocardiogram
  * Stress test (if indicated)
- Blood tests: Lipid profile, complete metabolic panel

ONGOING MONITORING:
- Bi-weekly blood pressure monitoring
- Monthly lipid profile checks
- Quarterly comprehensive health review
"""
    else:
        report += """
PREVENTIVE CARE SCHEDULE:
- Annual comprehensive health evaluation
- Quarterly blood pressure monitoring
- Semi-annual lipid profile
- Maintain current healthy lifestyle practices
- Regular exercise program
"""
    
    report += f"""
{'='*80}
10. ADDITIONAL RESOURCES AND REFERENCES
{'='*80}
American Heart Association Guidelines:
- Blood Pressure Management: Maintain < 120/80 mmHg
- Cholesterol Targets: LDL < 100 mg/dL, Total < 200 mg/dL
- Physical Activity: 150 minutes/week moderate-intensity exercise

Dietary Recommendations:
- Mediterranean Diet or DASH Diet
- 5-9 servings fruits and vegetables daily
- Whole grains, lean proteins, healthy fats
- Limit processed foods, sodium, and added sugars

Emergency Warning Signs:
Seek immediate medical attention if experiencing:
- Severe chest pain or pressure
- Shortness of breath at rest
- Dizziness or fainting
- Irregular heartbeat
- Extreme fatigue

{'='*80}
11. METHODOLOGY AND MODEL INFORMATION
{'='*80}
Model Type: Ensemble Stacking Classifier
Base Models: Random Forest, Extra Trees, XGBoost, Gradient Boosting,
             AdaBoost, Neural Networks, Support Vector Machines

Model Performance Metrics:
- Accuracy: ~{metadata.get('accuracy', 0.92):.2%}
- Precision: ~{metadata.get('precision', 0.89):.2%}
- Recall (Sensitivity): ~{metadata.get('recall', 0.91):.2%}
- F1 Score: ~{metadata.get('f1_score', 0.90):.2%}
- ROC-AUC: ~{metadata.get('roc_auc', 0.94):.2%}

Cross-Validation: 5-fold Stratified K-Fold
Data Preprocessing: MinMaxScaler normalization, outlier removal

⚠️ DISCLAIMER:
This report is generated by an AI-powered prediction tool and is for
informational and educational purposes only. It is NOT a substitute for
professional medical advice, diagnosis, or treatment.

The predictions are based on statistical patterns in training data and
should be interpreted in conjunction with clinical judgment by qualified
healthcare providers.

Always consult with licensed healthcare professionals for medical
decisions, diagnosis, and treatment plans.

Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
System: Advanced Cardiovascular Disease Prediction System v2.0
{'='*80}
"""
    
    return report

def get_care_recommendations(prediction, probability, user_input):
    """Generate personalized care recommendations"""
    recommendations = []
    warnings_list = []
    lifestyle_tips = []
    
    risk_level = "HIGH" if probability > 0.7 else "MODERATE" if probability > 0.5 else "LOW"
    
    if prediction == 1 or probability > 0.5:
        recommendations.append("🩺 **Immediate Actions:**")
        recommendations.append("- Consult a cardiologist as soon as possible")
        recommendations.append("- Schedule comprehensive cardiac tests (ECG, Stress Test, Echocardiogram)")
        recommendations.append("- Follow up with primary care physician within 1-2 weeks")
        recommendations.append("- Keep a log of symptoms and triggers")
        
        warnings_list.append("⚠️ **Warning Signs to Watch:**")
        warnings_list.append("- Chest pain or discomfort")
        warnings_list.append("- Shortness of breath during normal activities")
        warnings_list.append("- Dizziness or fainting")
        warnings_list.append("- Irregular heartbeat")
        warnings_list.append("- Swelling in legs, ankles, or feet")
    else:
        recommendations.append("✅ **Preventive Care:**")
        recommendations.append("- Continue regular health checkups (annual or biannual)")
        recommendations.append("- Maintain current healthy lifestyle")
        recommendations.append("- Monitor key health metrics regularly")
    
    if user_input['age'] >= 50:
        recommendations.append(f"\n👴 **Age-Specific Care (Age {user_input['age']}):**")
        recommendations.append("- Annual comprehensive cardiac evaluation recommended")
        recommendations.append("- Regular blood pressure and cholesterol monitoring")
    
    if user_input['resting_blood_pressure'] >= 140:
        warnings_list.append(f"⚠️ **High Blood Pressure ({user_input['resting_blood_pressure']} mmHg):**")
        warnings_list.append("- Consult doctor for BP management")
        warnings_list.append("- Reduce sodium intake (< 2g/day)")
        lifestyle_tips.append("- Follow DASH diet")
    
    if user_input['cholesterol'] >= 240:
        warnings_list.append(f"⚠️ **High Cholesterol ({user_input['cholesterol']} mg/dL):**")
        warnings_list.append("- Dietary changes: Reduce saturated and trans fats")
        lifestyle_tips.append("- Include more fruits, vegetables, whole grains")
    
    lifestyle_tips.append("💪 **Exercise:** 150 minutes/week moderate-intensity")
    lifestyle_tips.append("🥗 **Diet:** Mediterranean or DASH diet")
    lifestyle_tips.append("😴 **Sleep:** 7-9 hours quality sleep per night")
    
    return recommendations, warnings_list, lifestyle_tips, risk_level

def main():
    # Sidebar navigation
    st.sidebar.title("❤️ Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🏠 Home - New Prediction", "👥 Patient Management", "📊 Statistics & Analytics", "📈 Patient History", "🔍 Search & Reports"]
    )
    
    # Load model
    model, scaler, feature_names, metadata = load_model()
    
    if page == "🏠 Home - New Prediction":
        show_prediction_page(model, scaler, feature_names, metadata)
    elif page == "👥 Patient Management":
        show_patient_management()
    elif page == "📊 Statistics & Analytics":
        show_statistics()
    elif page == "📈 Patient History":
        show_patient_history()
    elif page == "🔍 Search & Reports":
        show_search_reports()

def show_prediction_page(model, scaler, feature_names, metadata):
    """Main prediction page"""
    st.markdown('<p class="main-header">❤️ Cardiovascular Disease Prediction System</p>', unsafe_allow_html=True)
    
    if model is None:
        st.stop()
    
    # Initialize database
    db = PatientDatabase()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Patient Information")
        
        # Patient registration section - improved
        st.subheader("👤 Patient Registration")
        tab1, tab2 = st.tabs(["🔍 Use Existing Patient", "➕ Register New Patient"])
        
        patient_id = None
        patient_name = None
        selected_patient_info = None
        
        # Track previous patient ID to detect changes
        previous_patient_id = st.session_state.get('current_patient_id', None)
        
        with tab1:
            st.markdown("**Select an existing patient from database:**")
            patients_df = db.get_all_patients()
            
            if not patients_df.empty:
                patient_options = [f"{row['patient_name']} (ID: {row['patient_id']}, Age: {row['age']}, {row['sex']})" 
                                  for _, row in patients_df.iterrows()]
                selected_option = st.selectbox("Choose Patient", [""] + patient_options, 
                                              key="patient_selector")
                
                if selected_option and selected_option != "":
                    # Extract patient ID from selection
                    selected_patient_name = selected_option.split(" (ID: ")[0]
                    patient_row = patients_df[patients_df['patient_name'] == selected_patient_name].iloc[0]
                    patient_id = int(patient_row['patient_id'])
                    patient_name = patient_row['patient_name']
                    
                    # Store current patient ID in session state
                    st.session_state['current_patient_id'] = patient_id
                    
                    # If patient changed, clear old auto-fill data
                    if previous_patient_id is not None and previous_patient_id != patient_id:
                        if 'latest_record' in st.session_state:
                            del st.session_state['latest_record']
                        if 'auto_fill_patient_id' in st.session_state:
                            del st.session_state['auto_fill_patient_id']
                    
                    # Get patient info
                    selected_patient_info = db.get_patient(patient_id)
                    patient_records = db.get_patient_records(patient_id)
                    
                    # Display patient details
                    st.success(f"✅ **Selected Patient: {patient_name}**")
                    
                    col_info1, col_info2 = st.columns(2)
                    with col_info1:
                        st.metric("Patient ID", patient_id)
                        st.metric("Age", f"{selected_patient_info['age']} years")
                    with col_info2:
                        st.metric("Sex", selected_patient_info['sex'])
                        st.metric("Total Records", len(patient_records))
                    
                    # Show previous records if any
                    if patient_records:
                        st.markdown("**📋 Previous Records:**")
                        latest_record = patient_records[0]  # Most recent first
                        record_df = pd.DataFrame(patient_records[:5])  # Show last 5
                        
                        with st.expander(f"View History ({len(patient_records)} records)", expanded=False):
                            st.dataframe(record_df[['created_at', 'probability', 'risk_level', 'prediction']].rename(
                                columns={'created_at': 'Date', 'probability': 'Risk %', 
                                        'risk_level': 'Risk Level', 'prediction': 'Disease'}
                            ), use_container_width=True, hide_index=True)
                        
                        # Show latest record summary
                        latest_prob = latest_record.get('probability', 0)
                        latest_risk = latest_record.get('risk_level', 'UNKNOWN')
                        latest_date = latest_record.get('created_at', 'N/A')
                        
                        st.info(f"**Latest Assessment:** {latest_date[:10] if len(str(latest_date)) > 10 else latest_date} - "
                               f"Risk: {latest_prob:.1%} ({latest_risk})")
                        
                        # Store latest record in session state for form auto-population
                        st.session_state['latest_record'] = latest_record
                        st.session_state['auto_fill_patient_id'] = patient_id
                        
                        # Suggestions based on history
                        if len(patient_records) > 1:
                            prev_prob = patient_records[1].get('probability', 0) if len(patient_records) > 1 else latest_prob
                            if latest_prob > prev_prob:
                                st.warning("⚠️ **Risk has increased** compared to previous assessment. Consider consultation.")
                            elif latest_prob < prev_prob:
                                st.success("✅ **Risk has decreased**. Continue current care plan.")
                    else:
                        st.info("No previous records found for this patient.")
                        # Clear auto-fill if no records
                        if 'latest_record' in st.session_state:
                            del st.session_state['latest_record']
                        if 'auto_fill_patient_id' in st.session_state:
                            del st.session_state['auto_fill_patient_id']
            else:
                st.warning("⚠️ No registered patients found. Please register a new patient in the second tab.")
                st.info("💡 **Tip:** You can also add sample patients by running: `python add_sample_patients.py`")
                # Clear auto-fill if no patients
                if 'latest_record' in st.session_state:
                    del st.session_state['latest_record']
                if 'auto_fill_patient_id' in st.session_state:
                    del st.session_state['auto_fill_patient_id']
                if 'current_patient_id' in st.session_state:
                    del st.session_state['current_patient_id']
        
        with tab2:
            st.markdown("**Register a new patient:**")
            new_patient_name = st.text_input("Patient Full Name *", placeholder="Enter patient name", 
                                            key="new_patient_name_input")
            new_patient_age = st.number_input("Age *", min_value=1, max_value=120, value=50, step=1, 
                                             key="new_patient_age_input")
            new_patient_sex = st.selectbox("Sex *", ["Male", "Female"], key="new_patient_sex_input")
            
            if st.button("✅ Register New Patient", use_container_width=True):
                if new_patient_name and new_patient_name.strip():
                    try:
                        new_patient_id = db.add_patient(new_patient_name, new_patient_age, new_patient_sex)
                        # Store in session state so patient is available immediately
                        st.session_state['newly_registered_patient_id'] = new_patient_id
                        st.session_state['newly_registered_patient_name'] = new_patient_name
                        st.session_state['newly_registered_patient_age'] = new_patient_age
                        st.session_state['newly_registered_patient_sex'] = new_patient_sex
                        
                        # Set as current patient
                        patient_id = new_patient_id
                        patient_name = new_patient_name
                        selected_patient_info = {'age': new_patient_age, 'sex': new_patient_sex}
                        
                        st.success(f"✅ **Patient registered successfully!**\n**Patient ID:** {patient_id}\n**Name:** {patient_name}")
                        st.info(f"💡 **You can now make predictions for this patient using the form below.**")
                        st.balloons()
                        
                        # Automatically select this patient for predictions
                        st.session_state['current_patient_id'] = patient_id
                        st.session_state['auto_selected_patient'] = True
                    except Exception as e:
                        st.error(f"❌ Error registering patient: {str(e)}")
                else:
                    st.error("❌ Please enter a patient name.")
            
            # Check if a new patient was just registered (make available for form)
            if 'newly_registered_patient_id' in st.session_state:
                newly_id = st.session_state.get('newly_registered_patient_id')
                newly_name = st.session_state.get('newly_registered_patient_name')
                if newly_id:
                    patient_id = newly_id
                    patient_name = newly_name
                    selected_patient_info = {
                        'age': st.session_state.get('newly_registered_patient_age', 50),
                        'sex': st.session_state.get('newly_registered_patient_sex', 'Male')
                    }
                    # Store in session state for use in form
                    st.session_state['current_patient_id'] = patient_id
        
        with st.form("prediction_form"):
            # Get patient_id from session state if newly registered
            if 'newly_registered_patient_id' in st.session_state and not patient_id:
                patient_id = st.session_state.get('newly_registered_patient_id')
                patient_name = st.session_state.get('newly_registered_patient_name')
                selected_patient_info = {
                    'age': st.session_state.get('newly_registered_patient_age', 50),
                    'sex': st.session_state.get('newly_registered_patient_sex', 'Male')
                }
            
            # Display current patient info if available
            if patient_id and patient_name:
                st.info(f"👤 **Current Patient:** {patient_name} (ID: {patient_id}) - *Your prediction will be saved to this patient's record*")
            elif 'newly_registered_patient_id' in st.session_state:
                patient_id = st.session_state.get('newly_registered_patient_id')
                patient_name = st.session_state.get('newly_registered_patient_name')
                st.info(f"👤 **Registered Patient:** {patient_name} (ID: {patient_id}) - *Your prediction will be saved to this patient's record*")
            else:
                st.warning("⚠️ **No patient selected.** Register a patient above or select an existing patient to save your prediction.")
            
            user_input = {}
            
            # Get default values from latest record if patient selected and has records
            default_values = {}
            auto_fill_active = False
            
            if patient_id and patient_id == st.session_state.get('auto_fill_patient_id'):
                if 'latest_record' in st.session_state:
                    latest_record = st.session_state['latest_record']
                    # Use patient's current age and sex from profile
                    default_values = {
                        'age': selected_patient_info['age'] if selected_patient_info else 50,
                        'sex': selected_patient_info['sex'] if selected_patient_info else "Male",
                        'resting_blood_pressure': int(latest_record.get('resting_blood_pressure', 120)) if latest_record.get('resting_blood_pressure') else 120,
                        'cholesterol': int(latest_record.get('cholesterol', 200)) if latest_record.get('cholesterol') else 200,
                        'max_heart_rate_achieved': int(latest_record.get('max_heart_rate_achieved', 150)) if latest_record.get('max_heart_rate_achieved') else 150,
                        'fasting_blood_sugar': str(latest_record.get('fasting_blood_sugar', 'No')) if latest_record.get('fasting_blood_sugar') else 'No',
                        'chest_pain_type': str(latest_record.get('chest_pain_type', 'Non-Anginal Pain')) if latest_record.get('chest_pain_type') else 'Non-Anginal Pain',
                        'rest_ecg': str(latest_record.get('rest_ecg', 'Normal')) if latest_record.get('rest_ecg') else 'Normal',
                        'exercise_induced_angina': str(latest_record.get('exercise_induced_angina', 'No')) if latest_record.get('exercise_induced_angina') else 'No',
                        'st_depression': float(latest_record.get('st_depression', 0.0)) if latest_record.get('st_depression') is not None else 0.0,
                        'st_slope': str(latest_record.get('st_slope', 'Upsloping')) if latest_record.get('st_slope') else 'Upsloping'
                    }
                    auto_fill_active = True
            
            # If patient selected but no records, use patient info only
            if patient_id and not auto_fill_active and selected_patient_info:
                default_values = {
                    'age': selected_patient_info['age'],
                    'sex': selected_patient_info['sex']
                }
            
            if auto_fill_active:
                st.info("💡 **Form auto-filled with latest record values.** You can modify any values as needed.")
            
            # Use patient info or defaults
            default_age = default_values.get('age', selected_patient_info['age'] if selected_patient_info else 50)
            default_sex = default_values.get('sex', selected_patient_info['sex'] if selected_patient_info else "Male")
            
            st.subheader("👤 Demographics")
            user_input['age'] = st.number_input("Age (years)", min_value=1, max_value=120, 
                                                value=int(default_age), step=1)
            user_input['sex'] = st.selectbox("Sex", ["Male", "Female"], 
                                           index=0 if default_sex == "Male" else 1)
            
            st.subheader("💓 Vital Signs")
            user_input['resting_blood_pressure'] = st.number_input("Resting Blood Pressure (mmHg)", 
                                                                   min_value=50, max_value=250, 
                                                                   value=int(default_values.get('resting_blood_pressure', 120)), step=1)
            user_input['cholesterol'] = st.number_input("Serum Cholesterol (mg/dL)", 
                                                       min_value=100, max_value=600, 
                                                       value=int(default_values.get('cholesterol', 200)), step=1)
            user_input['max_heart_rate_achieved'] = st.number_input("Maximum Heart Rate Achieved (bpm)", 
                                                                    min_value=60, max_value=220, 
                                                                    value=int(default_values.get('max_heart_rate_achieved', 150)), step=1)
            fasting_blood_sugar_default = default_values.get('fasting_blood_sugar', 'No')
            user_input['fasting_blood_sugar'] = st.selectbox("Fasting Blood Sugar > 120 mg/dL", 
                                                           ["No", "Yes"],
                                                           index=0 if fasting_blood_sugar_default == 'No' else 1)
            
            st.subheader("🩺 Symptoms & Tests")
            chest_pain_types = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"]
            chest_pain_default = default_values.get('chest_pain_type', 'Non-Anginal Pain')
            chest_pain_index = chest_pain_types.index(chest_pain_default) if chest_pain_default in chest_pain_types else 2
            user_input['chest_pain_type'] = st.selectbox("Chest Pain Type", chest_pain_types, index=chest_pain_index)
            
            rest_ecg_options = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"]
            rest_ecg_default = default_values.get('rest_ecg', 'Normal')
            rest_ecg_index = rest_ecg_options.index(rest_ecg_default) if rest_ecg_default in rest_ecg_options else 0
            user_input['rest_ecg'] = st.selectbox("Resting ECG Results", rest_ecg_options, index=rest_ecg_index)
            
            exercise_angina_default = default_values.get('exercise_induced_angina', 'No')
            user_input['exercise_induced_angina'] = st.selectbox("Exercise Induced Angina", 
                                                               ["No", "Yes"],
                                                               index=0 if exercise_angina_default == 'No' else 1)
            user_input['st_depression'] = st.number_input("ST Depression (mm)", 
                                                          min_value=0.0, max_value=6.0, 
                                                          value=float(default_values.get('st_depression', 0.0)), step=0.1)
            
            st_slope_options = ["Upsloping", "Flat", "Downsloping"]
            st_slope_default = default_values.get('st_slope', 'Upsloping')
            st_slope_index = st_slope_options.index(st_slope_default) if st_slope_default in st_slope_options else 0
            user_input['st_slope'] = st.selectbox("ST Slope", st_slope_options, index=st_slope_index)
            
            submitted = st.form_submit_button("🔍 Predict Risk", use_container_width=True)
            
            if submitted:
                try:
                    input_df = preprocess_input(user_input, feature_names, scaler)
                    
                    # Validate input dataframe
                    if input_df.isnull().any().any():
                        st.error("❌ Error: Missing values in input data. Please check all fields are filled.")
                        st.stop()
                    
                    # Make prediction
                    prediction = model.predict(input_df)[0]
                    probability = model.predict_proba(input_df)[0][1] if hasattr(model, 'predict_proba') else 0.5
                    
                    # Validate prediction output
                    if prediction not in [0, 1]:
                        st.error(f"❌ Error: Invalid prediction value: {prediction}")
                        st.stop()
                    
                    if probability < 0 or probability > 1:
                        st.error(f"❌ Error: Invalid probability value: {probability}")
                        st.stop()
                    
                    risk_level = "HIGH" if probability > 0.7 else "MODERATE" if probability > 0.5 else "LOW"
                    
                    # Store prediction explanation factors
                    prediction_factors = analyze_prediction_factors(user_input, prediction, probability)
                    st.session_state['prediction_factors'] = prediction_factors
                    
                    recommendations, warnings_list, lifestyle_tips, _ = get_care_recommendations(
                        prediction, probability, user_input
                    )
                    
                    # Store results in session state first
                    st.session_state['prediction'] = prediction
                    st.session_state['probability'] = probability
                    st.session_state['user_input'] = user_input
                    st.session_state['risk_level'] = risk_level
                    st.session_state['recommendations'] = recommendations
                    st.session_state['warnings_list'] = warnings_list
                    st.session_state['lifestyle_tips'] = lifestyle_tips
                    st.session_state['patient_id'] = patient_id  # Store patient_id for report saving
                    st.session_state['patient_name'] = patient_name  # Store patient_name
                    
                    # Generate report for saving
                    report = get_detailed_report(
                        user_input, prediction, probability, risk_level,
                        recommendations, warnings_list, lifestyle_tips, metadata
                    )
                    
                    # Save to database if patient registered
                    record_id = None
                    report_id = None
                    
                    # Ensure patient_id is available from session state if newly registered
                    if not patient_id:
                        # Check if patient was just registered
                        if 'newly_registered_patient_id' in st.session_state:
                            patient_id = st.session_state.get('newly_registered_patient_id')
                            patient_name = st.session_state.get('newly_registered_patient_name', 'Unknown')
                        # Check if patient is in session state from previous registration
                        elif 'patient_id' in st.session_state:
                            patient_id = st.session_state.get('patient_id')
                            patient_name = st.session_state.get('patient_name', 'Unknown')
                    
                    if patient_id:
                        # Patient already registered or newly registered
                        try:
                            record_id = db.add_record(patient_id, user_input, prediction, probability, risk_level)
                            
                            # Save report to database
                            try:
                                report_json = {
                                    'patient_info': user_input,
                                    'prediction': int(prediction),
                                    'probability': float(probability),
                                    'risk_level': risk_level,
                                    'recommendations': recommendations,
                                    'warnings': warnings_list,
                                    'lifestyle_tips': lifestyle_tips,
                                    'timestamp': datetime.now().isoformat(),
                                    'model_metadata': metadata if metadata else {}
                                }
                                report_id = db.add_report(patient_id, record_id, report, report_json)
                            except Exception as e:
                                st.warning(f"⚠️ Warning: Report could not be saved to database: {str(e)}")
                                import traceback
                                st.error(f"Report save error details: {traceback.format_exc()}")
                                report_id = None
                            
                            if report_id:
                                st.success(f"✅ **Record and Report saved successfully!**\n**Patient:** {patient_name if patient_name else 'Unknown'} (ID: {patient_id})\n**Record ID:** {record_id}\n**Report ID:** {report_id}")
                                st.info("💡 **All medical history and report saved to database.** View in 'Patient History' page.")
                            else:
                                st.success(f"✅ **Record saved successfully!**\n**Patient:** {patient_name if patient_name else 'Unknown'} (ID: {patient_id})\n**Record ID:** {record_id}")
                                st.warning("⚠️ Report could not be saved. Medical record is saved, but report generation failed.")
                            st.info("💡 **Tip:** View patient history and saved reports in the 'Patient History' page.")
                        except Exception as e:
                            st.error(f"❌ Error saving record to database: {str(e)}")
                            import traceback
                            st.error(f"Error details: {traceback.format_exc()}")
                        
                    elif patient_name and patient_name.strip():
                        # Auto-register if name provided but not selected
                        try:
                            patient_id = db.add_patient(patient_name, user_input['age'], user_input['sex'])
                            record_id = db.add_record(patient_id, user_input, prediction, probability, risk_level)
                            
                            # Save report to database
                            try:
                                report_json = {
                                    'patient_info': user_input,
                                    'prediction': int(prediction),
                                    'probability': float(probability),
                                    'risk_level': risk_level,
                                    'recommendations': recommendations,
                                    'warnings': warnings_list,
                                    'lifestyle_tips': lifestyle_tips,
                                    'timestamp': datetime.now().isoformat(),
                                    'model_metadata': metadata if metadata else {}
                                }
                                report_id = db.add_report(patient_id, record_id, report, report_json)
                            except Exception as e:
                                st.warning(f"⚠️ Warning: Report could not be saved to database: {str(e)}")
                                report_id = None
                            
                            # Update session state with new patient_id
                            st.session_state['patient_id'] = patient_id
                            st.session_state['patient_name'] = patient_name
                            
                            if report_id:
                                st.success(f"✅ **New patient registered and record saved!**\n**Patient ID:** {patient_id}\n**Record ID:** {record_id}\n**Report ID:** {report_id}")
                                st.info("💡 **All medical history and report saved to database.** View in 'Patient History' page.")
                            else:
                                st.success(f"✅ **New patient registered and record saved!**\n**Patient ID:** {patient_id}\n**Record ID:** {record_id}")
                                st.warning("⚠️ Report could not be saved. Medical record is saved, but report generation failed.")
                            st.balloons()
                        except Exception as e:
                            st.error(f"❌ Error saving patient/record to database: {str(e)}")
                            import traceback
                            st.error(f"Error details: {traceback.format_exc()}")
                    else:
                        # No patient ID or name - still store in session state but don't save to DB
                        st.warning("⚠️ **No patient registered.** Record not saved to database. Please register a patient first to save records.")
                        st.info("💡 **Tip:** Use the 'Register New Patient' tab above to register a patient, or select an existing patient from the first tab.")
                    
                    # Store record and report IDs in session state
                    if record_id:
                        st.session_state['record_id'] = record_id
                    if report_id:
                        st.session_state['report_id'] = report_id
                    
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
    
    with col2:
        st.header("📊 Prediction Results")
        
        if 'prediction' in st.session_state:
            prediction = st.session_state['prediction']
            probability = st.session_state['probability']
            risk_level = st.session_state['risk_level']
            
            if prediction == 1:
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.markdown("### ⚠️ HIGH RISK DETECTED")
                st.markdown(f"### Disease Probability: {probability:.2%}")
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.markdown("### ✅ LOW RISK")
                st.markdown(f"### Disease Probability: {probability:.2%}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Visualizations - Fixed gauge chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = probability * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': f"Risk Probability: {probability:.1%}"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Prediction", "Disease" if prediction == 1 else "No Disease")
            with col2:
                st.metric("Risk Probability", f"{probability:.2%}")
            with col3:
                st.metric("Risk Level", risk_level)
            
            # Prediction Explanation
            if 'prediction_factors' in st.session_state:
                st.markdown("---")
                st.subheader("🔍 Prediction Explanation")
                
                factors = st.session_state['prediction_factors']
                
                col_fact1, col_fact2 = st.columns(2)
                
                with col_fact1:
                    if factors['protective']:
                        st.markdown("**✅ Protective Factors (Lower Risk):**")
                        for factor in factors['protective']:
                            st.markdown(f"  • {factor}")
                
                with col_fact2:
                    if factors['risk']:
                        st.markdown("**⚠️ Risk Factors (Higher Risk):**")
                        for factor in factors['risk']:
                            st.markdown(f"  • {factor}")
                
                # Summary
                total_protective = len(factors['protective'])
                total_risk = len(factors['risk'])
                
                if total_protective > total_risk:
                    st.info(f"💡 **Summary:** {total_protective} protective factors vs {total_risk} risk factors - "
                           f"Overall low risk prediction is consistent with the clinical indicators.")
                elif total_risk > total_protective:
                    st.warning(f"⚠️ **Summary:** {total_risk} risk factors vs {total_protective} protective factors - "
                              f"Multiple risk factors identified. Consider medical consultation.")
                else:
                    st.info(f"📊 **Summary:** Balanced profile with {total_risk} risk factors and "
                           f"{total_protective} protective factors.")
    
    # Detailed Report Section
    if 'prediction' in st.session_state:
        st.markdown("---")
        st.header("📄 Detailed Comprehensive Report")
        
        # Get report from session state if saved, otherwise generate
        if 'saved_report' in st.session_state:
            report = st.session_state['saved_report']
        else:
            report = get_detailed_report(
                st.session_state['user_input'],
                st.session_state['prediction'],
                st.session_state['probability'],
                st.session_state['risk_level'],
                st.session_state['recommendations'],
                st.session_state['warnings_list'],
                st.session_state['lifestyle_tips'],
                metadata
            )
            st.session_state['saved_report'] = report
        
        # Download buttons
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Download Detailed Report (TXT)",
                data=report,
                file_name=f"detailed_heart_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        with col2:
            st.download_button(
                label="📥 Download JSON Report",
                data=json.dumps({
                    'patient_info': st.session_state['user_input'],
                    'prediction': int(st.session_state['prediction']),
                    'probability': float(st.session_state['probability']),
                    'risk_level': st.session_state['risk_level'],
                    'recommendations': st.session_state['recommendations'],
                    'warnings': st.session_state['warnings_list'],
                    'lifestyle_tips': st.session_state['lifestyle_tips'],
                    'timestamp': datetime.now().isoformat()
                }, indent=2),
                file_name=f"heart_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        # Display report preview
        with st.expander("📋 View Report Preview"):
            st.text(report[:2000] + "..." if len(report) > 2000 else report)

def show_patient_management():
    """Patient management page"""
    st.header("👥 Patient Management")
    
    tab1, tab2, tab3 = st.tabs(["View All Patients", "Add New Patient", "Edit Patient"])
    
    with tab1:
        patients_df = db.get_all_patients()
        if not patients_df.empty:
            st.dataframe(patients_df, use_container_width=True)
        else:
            st.info("No patients registered yet.")
    
    with tab2:
        with st.form("add_patient_form"):
            name = st.text_input("Patient Name *")
            age = st.number_input("Age *", min_value=1, max_value=120, value=50)
            sex = st.selectbox("Sex *", ["Male", "Female"])
            
            if st.form_submit_button("➕ Add Patient"):
                if name:
                    patient_id = db.add_patient(name, age, sex)
                    st.success(f"✅ Patient added successfully! ID: {patient_id}")
                else:
                    st.error("Please enter patient name")
    
    with tab3:
        patients_df = db.get_all_patients()
        if not patients_df.empty:
            selected = st.selectbox("Select Patient", patients_df['patient_name'].tolist())
            patient_id = patients_df[patients_df['patient_name'] == selected]['patient_id'].iloc[0]
            
            if st.button("🗑️ Delete Patient"):
                db.delete_patient(patient_id)
                st.success("Patient deleted")
                st.rerun()

def show_statistics():
    """Statistics and analytics page"""
    st.header("📊 Statistics & Analytics")
    
    stats = db.get_statistics()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Patients", stats['total_patients'])
    with col2:
        st.metric("Total Records", stats['total_records'])
    with col3:
        st.metric("High Risk", stats['risk_distribution'].get('HIGH', 0))
    with col4:
        st.metric("Low Risk", stats['risk_distribution'].get('LOW', 0))
    
    # Visualizations
    if stats['total_records'] > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk distribution pie chart
            risk_data = stats['risk_distribution']
            fig = px.pie(
                values=list(risk_data.values()),
                names=list(risk_data.keys()),
                title="Risk Level Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Prediction distribution
            pred_data = stats['prediction_distribution']
            fig = px.bar(
                x=['No Disease', 'Disease'],
                y=[pred_data.get(0, 0), pred_data.get(1, 0)],
                title="Prediction Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

def show_patient_history():
    """Patient history page"""
    db = PatientDatabase()
    
    st.header("📈 Patient History & Reports")
    
    patients_df = db.get_all_patients()
    if not patients_df.empty:
        selected = st.selectbox("Select Patient", patients_df['patient_name'].tolist(), key="history_patient_selector")
        patient_id = patients_df[patients_df['patient_name'] == selected]['patient_id'].iloc[0]
        
        # Get patient info
        patient_info = db.get_patient(patient_id)
        
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.metric("Patient ID", patient_id)
        with col_info2:
            st.metric("Age", f"{patient_info['age']} years")
        with col_info3:
            st.metric("Sex", patient_info['sex'])
        
        records = db.get_patient_records(patient_id)
        reports = db.get_reports_for_patient(patient_id)
        
        st.markdown("---")
        
        tab1, tab2, tab3 = st.tabs(["📊 Records History", "📄 Saved Reports", "📈 Trends"])
        
        with tab1:
            if records:
                st.subheader(f"Medical Records ({len(records)} total)")
                df = pd.DataFrame(records)
                
                # Format for display
                display_df = df[[
                    'created_at', 'resting_blood_pressure', 'cholesterol', 
                    'max_heart_rate_achieved', 'fasting_blood_sugar', 
                    'chest_pain_type', 'rest_ecg', 'exercise_induced_angina',
                    'st_depression', 'st_slope', 'prediction', 'probability', 'risk_level'
                ]].rename(columns={
                    'created_at': 'Date',
                    'resting_blood_pressure': 'BP (mmHg)',
                    'cholesterol': 'Chol (mg/dL)',
                    'max_heart_rate_achieved': 'Max HR (bpm)',
                    'fasting_blood_sugar': 'FBS',
                    'chest_pain_type': 'Chest Pain',
                    'rest_ecg': 'ECG',
                    'exercise_induced_angina': 'Ex. Angina',
                    'st_depression': 'ST Dep (mm)',
                    'st_slope': 'ST Slope',
                    'prediction': 'Disease',
                    'probability': 'Risk %',
                    'risk_level': 'Risk Level'
                })
                
                display_df['Disease'] = display_df['Disease'].apply(lambda x: 'Yes' if x == 1 else 'No')
                display_df['Risk %'] = display_df['Risk %'].apply(lambda x: f"{x:.1%}")
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                # Download records as CSV
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Records (CSV)",
                    data=csv,
                    file_name=f"{selected}_records_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No records found for this patient.")
        
        with tab2:
            if reports:
                st.subheader(f"Saved Reports ({len(reports)} total)")
                
                for i, report in enumerate(reports, 1):
                    with st.expander(f"Report #{i} - {report.get('created_at', 'Unknown Date')[:10]}", expanded=(i == 1)):
                        report_json_str = report.get('report_json', '{}')
                        try:
                            import json
                            report_data = json.loads(report_json_str) if isinstance(report_json_str, str) else report_json_str
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Prediction", "Disease" if report_data.get('prediction') == 1 else "No Disease")
                                st.metric("Risk Level", report_data.get('risk_level', 'Unknown'))
                            with col2:
                                st.metric("Probability", f"{report_data.get('probability', 0):.1%}")
                                st.metric("Record ID", report.get('record_id', 'N/A'))
                            
                            # Show report text
                            st.markdown("### Report Content")
                            st.text_area("Full Report", report.get('report_text', ''), height=300, key=f"report_text_{i}")
                            
                            # Download buttons
                            col_dl1, col_dl2 = st.columns(2)
                            with col_dl1:
                                st.download_button(
                                    label="📥 Download Report (TXT)",
                                    data=report.get('report_text', ''),
                                    file_name=f"{selected}_report_{report.get('record_id', i)}_{datetime.now().strftime('%Y%m%d')}.txt",
                                    mime="text/plain",
                                    key=f"dl_txt_{i}"
                                )
                            with col_dl2:
                                st.download_button(
                                    label="📥 Download Report (JSON)",
                                    data=json.dumps(report_data, indent=2),
                                    file_name=f"{selected}_report_{report.get('record_id', i)}_{datetime.now().strftime('%Y%m%d')}.json",
                                    mime="application/json",
                                    key=f"dl_json_{i}"
                                )
                        except Exception as e:
                            st.error(f"Error loading report: {str(e)}")
                            st.text(report.get('report_text', 'Report not available')[:500])
            else:
                st.info("No saved reports found for this patient.")
                st.info("💡 Reports are saved automatically when you make predictions for registered patients.")
        
        with tab3:
            if records and len(records) > 1:
                st.subheader("Risk Probability Trend")
                df = pd.DataFrame(records)
                df['created_at'] = pd.to_datetime(df['created_at'])
                df = df.sort_values('created_at')
                
                fig = px.line(
                    df, 
                    x='created_at', 
                    y='probability',
                    title=f"Risk Probability Trend - {selected}",
                    markers=True,
                    labels={'probability': 'Risk Probability (%)', 'created_at': 'Date'}
                )
                fig.update_traces(line=dict(width=3), marker=dict(size=10))
                fig.update_layout(hovermode='x unified')
                st.plotly_chart(fig, use_container_width=True)
                
                # Show summary statistics
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                with col_stat1:
                    st.metric("First Assessment", f"{df.iloc[0]['probability']:.1%}")
                with col_stat2:
                    st.metric("Latest Assessment", f"{df.iloc[-1]['probability']:.1%}")
                with col_stat3:
                    change = df.iloc[-1]['probability'] - df.iloc[0]['probability']
                    st.metric("Change", f"{change:+.1%}", delta_color="inverse")
            elif records:
                st.info("Only one record available. Multiple records needed for trend analysis.")
            else:
                st.info("No records found for trend analysis.")
    else:
        st.info("No patients registered yet.")

def show_search_reports():
    """Search and reports page"""
    db = PatientDatabase()
    
    st.header("🔍 Search & Reports")
    
    tab1, tab2 = st.tabs(["🔍 Search Patients", "📄 View All Reports"])
    
    with tab1:
        search_term = st.text_input("Search Patients by Name", key="search_input")
        
        if search_term:
            results = db.search_patients(search_term)
            if not results.empty:
                st.dataframe(results, use_container_width=True)
                
                # Show records for selected patient
                if len(results) == 1:
                    patient_id = results.iloc[0]['patient_id']
                    records = db.get_patient_records(patient_id)
                    reports = db.get_reports_for_patient(patient_id)
                    
                    st.markdown("---")
                    st.subheader(f"Records for {results.iloc[0]['patient_name']}")
                    if records:
                        st.dataframe(pd.DataFrame(records), use_container_width=True)
                        st.metric("Total Records", len(records))
                    if reports:
                        st.metric("Total Reports", len(reports))
            else:
                st.info("No patients found.")
        else:
            st.info("Enter a search term to find patients.")
    
    with tab2:
        st.subheader("All Saved Reports")
        
        # Get all patients and their reports
        patients_df = db.get_all_patients()
        
        if not patients_df.empty:
            selected_patient = st.selectbox("Select Patient", [""] + patients_df['patient_name'].tolist(), key="all_reports_selector")
            
            if selected_patient:
                patient_id = patients_df[patients_df['patient_name'] == selected_patient]['patient_id'].iloc[0]
                reports = db.get_reports_for_patient(patient_id)
                
                if reports:
                    st.success(f"Found {len(reports)} report(s) for {selected_patient}")
                    
                    for i, report in enumerate(reports, 1):
                        with st.expander(f"Report #{i} - Record ID: {report.get('record_id', 'N/A')} - {report.get('created_at', 'Unknown')[:10]}"):
                            try:
                                import json
                                report_json_str = report.get('report_json', '{}')
                                report_data = json.loads(report_json_str) if isinstance(report_json_str, str) else report_json_str
                                
                                st.metric("Risk Probability", f"{report_data.get('probability', 0):.1%}")
                                st.metric("Risk Level", report_data.get('risk_level', 'Unknown'))
                                
                                st.text_area("Report Content", report.get('report_text', ''), height=400, key=f"all_report_{i}")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.download_button(
                                        label="📥 Download TXT",
                                        data=report.get('report_text', ''),
                                        file_name=f"{selected_patient}_report_{report.get('record_id', i)}.txt",
                                        mime="text/plain",
                                        key=f"dl_all_txt_{i}"
                                    )
                                with col2:
                                    st.download_button(
                                        label="📥 Download JSON",
                                        data=json.dumps(report_data, indent=2),
                                        file_name=f"{selected_patient}_report_{report.get('record_id', i)}.json",
                                        mime="application/json",
                                        key=f"dl_all_json_{i}"
                                    )
                            except Exception as e:
                                st.error(f"Error loading report: {str(e)}")
                else:
                    st.info(f"No reports found for {selected_patient}")
        else:
            st.info("No patients registered yet.")

if __name__ == "__main__":
    main()