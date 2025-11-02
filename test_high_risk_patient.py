"""
Script to add a high-risk test patient to the database
This patient will demonstrate HIGH RISK prediction values
"""

from patient_database import PatientDatabase

def add_high_risk_test_patient():
    """Add a high-risk test patient"""
    db = PatientDatabase()
    
    # High-risk patient profile
    patient_name = "High Risk Test Patient"
    age = 58
    sex = "Male"
    
    # High-risk record values
    high_risk_record = {
        'resting_blood_pressure': 145,
        'cholesterol': 245,
        'fasting_blood_sugar': 'Yes',
        'max_heart_rate_achieved': 145,
        'chest_pain_type': 'Typical Angina',
        'rest_ecg': 'ST-T Wave Abnormality',
        'exercise_induced_angina': 'Yes',
        'st_depression': 2.5,
        'st_slope': 'Downsloping'
    }
    
    print("Adding high-risk test patient...")
    print("=" * 60)
    
    # Check if patient already exists
    patients_df = db.get_all_patients()
    existing = patients_df[patients_df['patient_name'] == patient_name]
    
    if not existing.empty:
        patient_id = int(existing.iloc[0]['patient_id'])
        print(f"Patient '{patient_name}' already exists (ID: {patient_id})")
        print("Adding new high-risk record...")
    else:
        # Add new patient
        patient_id = db.add_patient(patient_name, age, sex)
        print(f"[OK] Added patient: {patient_name} (ID: {patient_id})")
    
    # Add high-risk record with prediction (this is a test prediction)
    # Note: You'll need to run actual prediction through the model for real values
    # This is just for demonstration purposes
    prediction = 1  # HIGH RISK
    probability = 0.88  # 88% probability
    risk_level = "HIGH"
    
    record_id = db.add_record(
        patient_id,
        high_risk_record,
        prediction,
        probability,
        risk_level
    )
    
    print(f"[OK] Added high-risk record (ID: {record_id})")
    print("=" * 60)
    print("\nHigh-Risk Patient Values:")
    print(f"  Age: {age}")
    print(f"  Sex: {sex}")
    print(f"  BP: {high_risk_record['resting_blood_pressure']} mmHg")
    print(f"  Cholesterol: {high_risk_record['cholesterol']} mg/dL")
    print(f"  Max HR: {high_risk_record['max_heart_rate_achieved']} bpm")
    print(f"  Fasting BS: {high_risk_record['fasting_blood_sugar']}")
    print(f"  Chest Pain: {high_risk_record['chest_pain_type']}")
    print(f"  ECG: {high_risk_record['rest_ecg']}")
    print(f"  Exercise Angina: {high_risk_record['exercise_induced_angina']}")
    print(f"  ST Depression: {high_risk_record['st_depression']} mm")
    print(f"  ST Slope: {high_risk_record['st_slope']}")
    print("\n[SUCCESS] High-risk test patient added!")
    print("\n[INFO] To test HIGH RISK prediction:")
    print("   1. Run Streamlit app")
    print("   2. Select this patient or use these values manually")
    print("   3. Run prediction to see HIGH RISK result")

if __name__ == "__main__":
    add_high_risk_test_patient()
