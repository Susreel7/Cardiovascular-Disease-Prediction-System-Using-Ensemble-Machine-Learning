"""
Script to add sample patients to the database
"""

from patient_database import PatientDatabase
import random

# Sample patient data with realistic cardiovascular information
sample_patients = [
    {
        "name": "John Smith",
        "age": 58,
        "sex": "Male",
        "records": [
            {"resting_blood_pressure": 145, "cholesterol": 245, "fasting_blood_sugar": "Yes",
             "max_heart_rate_achieved": 145, "chest_pain_type": "Typical Angina",
             "rest_ecg": "ST-T Wave Abnormality", "exercise_induced_angina": "Yes",
             "st_depression": 2.5, "st_slope": "Downsloping", "prediction": 1, "probability": 0.85, "risk_level": "HIGH"}
        ]
    },
    {
        "name": "Mary Johnson",
        "age": 45,
        "sex": "Female",
        "records": [
            {"resting_blood_pressure": 118, "cholesterol": 185, "fasting_blood_sugar": "No",
             "max_heart_rate_achieved": 162, "chest_pain_type": "Non-Anginal Pain",
             "rest_ecg": "Normal", "exercise_induced_angina": "No",
             "st_depression": 0.5, "st_slope": "Upsloping", "prediction": 0, "probability": 0.28, "risk_level": "LOW"}
        ]
    },
    {
        "name": "Robert Williams",
        "age": 62,
        "sex": "Male",
        "records": [
            {"resting_blood_pressure": 138, "cholesterol": 220, "fasting_blood_sugar": "No",
             "max_heart_rate_achieved": 138, "chest_pain_type": "Atypical Angina",
             "rest_ecg": "ST-T Wave Abnormality", "exercise_induced_angina": "Yes",
             "st_depression": 1.8, "st_slope": "Flat", "prediction": 1, "probability": 0.72, "risk_level": "MODERATE"}
        ]
    },
    {
        "name": "Patricia Brown",
        "age": 52,
        "sex": "Female",
        "records": [
            {"resting_blood_pressure": 132, "cholesterol": 210, "fasting_blood_sugar": "Yes",
             "max_heart_rate_achieved": 150, "chest_pain_type": "Asymptomatic",
             "rest_ecg": "Normal", "exercise_induced_angina": "No",
             "st_depression": 1.0, "st_slope": "Upsloping", "prediction": 0, "probability": 0.42, "risk_level": "LOW"},
            {"resting_blood_pressure": 128, "cholesterol": 195, "fasting_blood_sugar": "No",
             "max_heart_rate_achieved": 155, "chest_pain_type": "Non-Anginal Pain",
             "rest_ecg": "Normal", "exercise_induced_angina": "No",
             "st_depression": 0.3, "st_slope": "Upsloping", "prediction": 0, "probability": 0.25, "risk_level": "LOW"}
        ]
    },
    {
        "name": "Michael Davis",
        "age": 55,
        "sex": "Male",
        "records": [
            {"resting_blood_pressure": 150, "cholesterol": 260, "fasting_blood_sugar": "Yes",
             "max_heart_rate_achieved": 140, "chest_pain_type": "Typical Angina",
             "rest_ecg": "Left Ventricular Hypertrophy", "exercise_induced_angina": "Yes",
             "st_depression": 3.0, "st_slope": "Downsloping", "prediction": 1, "probability": 0.92, "risk_level": "HIGH"}
        ]
    },
    {
        "name": "Linda Miller",
        "age": 48,
        "sex": "Female",
        "records": [
            {"resting_blood_pressure": 125, "cholesterol": 195, "fasting_blood_sugar": "No",
             "max_heart_rate_achieved": 168, "chest_pain_type": "Non-Anginal Pain",
             "rest_ecg": "Normal", "exercise_induced_angina": "No",
             "st_depression": 0.2, "st_slope": "Upsloping", "prediction": 0, "probability": 0.18, "risk_level": "LOW"}
        ]
    },
    {
        "name": "James Wilson",
        "age": 65,
        "sex": "Male",
        "records": [
            {"resting_blood_pressure": 142, "cholesterol": 230, "fasting_blood_sugar": "No",
             "max_heart_rate_achieved": 132, "chest_pain_type": "Atypical Angina",
             "rest_ecg": "ST-T Wave Abnormality", "exercise_induced_angina": "Yes",
             "st_depression": 1.5, "st_slope": "Flat", "prediction": 1, "probability": 0.68, "risk_level": "MODERATE"}
        ]
    },
    {
        "name": "Barbara Moore",
        "age": 50,
        "sex": "Female",
        "records": [
            {"resting_blood_pressure": 135, "cholesterol": 205, "fasting_blood_sugar": "No",
             "max_heart_rate_achieved": 158, "chest_pain_type": "Non-Anginal Pain",
             "rest_ecg": "Normal", "exercise_induced_angina": "No",
             "st_depression": 0.8, "st_slope": "Upsloping", "prediction": 0, "probability": 0.35, "risk_level": "LOW"}
        ]
    },
    {
        "name": "William Taylor",
        "age": 59,
        "sex": "Male",
        "records": [
            {"resting_blood_pressure": 148, "cholesterol": 255, "fasting_blood_sugar": "Yes",
             "max_heart_rate_achieved": 135, "chest_pain_type": "Typical Angina",
             "rest_ecg": "Left Ventricular Hypertrophy", "exercise_induced_angina": "Yes",
             "st_depression": 2.8, "st_slope": "Downsloping", "prediction": 1, "probability": 0.88, "risk_level": "HIGH"}
        ]
    },
    {
        "name": "Jennifer Anderson",
        "age": 43,
        "sex": "Female",
        "records": [
            {"resting_blood_pressure": 115, "cholesterol": 175, "fasting_blood_sugar": "No",
             "max_heart_rate_achieved": 175, "chest_pain_type": "Non-Anginal Pain",
             "rest_ecg": "Normal", "exercise_induced_angina": "No",
             "st_depression": 0.0, "st_slope": "Upsloping", "prediction": 0, "probability": 0.15, "risk_level": "LOW"}
        ]
    },
    {
        "name": "Richard Thomas",
        "age": 61,
        "sex": "Male",
        "records": [
            {"resting_blood_pressure": 140, "cholesterol": 225, "fasting_blood_sugar": "No",
             "max_heart_rate_achieved": 142, "chest_pain_type": "Atypical Angina",
             "rest_ecg": "ST-T Wave Abnormality", "exercise_induced_angina": "No",
             "st_depression": 1.2, "st_slope": "Flat", "prediction": 1, "probability": 0.58, "risk_level": "MODERATE"},
            {"resting_blood_pressure": 145, "cholesterol": 235, "fasting_blood_sugar": "No",
             "max_heart_rate_achieved": 138, "chest_pain_type": "Atypical Angina",
             "rest_ecg": "ST-T Wave Abnormality", "exercise_induced_angina": "Yes",
             "st_depression": 1.8, "st_slope": "Flat", "prediction": 1, "probability": 0.65, "risk_level": "MODERATE"}
        ]
    },
    {
        "name": "Susan Jackson",
        "age": 47,
        "sex": "Female",
        "records": [
            {"resting_blood_pressure": 122, "cholesterol": 190, "fasting_blood_sugar": "No",
             "max_heart_rate_achieved": 165, "chest_pain_type": "Non-Anginal Pain",
             "rest_ecg": "Normal", "exercise_induced_angina": "No",
             "st_depression": 0.4, "st_slope": "Upsloping", "prediction": 0, "probability": 0.22, "risk_level": "LOW"}
        ]
    },
    {
        "name": "David White",
        "age": 54,
        "sex": "Male",
        "records": [
            {"resting_blood_pressure": 143, "cholesterol": 250, "fasting_blood_sugar": "Yes",
             "max_heart_rate_achieved": 145, "chest_pain_type": "Typical Angina",
             "rest_ecg": "ST-T Wave Abnormality", "exercise_induced_angina": "Yes",
             "st_depression": 2.3, "st_slope": "Downsloping", "prediction": 1, "probability": 0.82, "risk_level": "HIGH"}
        ]
    },
    {
        "name": "Nancy Harris",
        "age": 51,
        "sex": "Female",
        "records": [
            {"resting_blood_pressure": 128, "cholesterol": 200, "fasting_blood_sugar": "No",
             "max_heart_rate_achieved": 160, "chest_pain_type": "Asymptomatic",
             "rest_ecg": "Normal", "exercise_induced_angina": "No",
             "st_depression": 0.6, "st_slope": "Upsloping", "prediction": 0, "probability": 0.32, "risk_level": "LOW"}
        ]
    },
    {
        "name": "Charles Martin",
        "age": 57,
        "sex": "Male",
        "records": [
            {"resting_blood_pressure": 146, "cholesterol": 240, "fasting_blood_sugar": "No",
             "max_heart_rate_achieved": 136, "chest_pain_type": "Atypical Angina",
             "rest_ecg": "Left Ventricular Hypertrophy", "exercise_induced_angina": "Yes",
             "st_depression": 2.1, "st_slope": "Flat", "prediction": 1, "probability": 0.75, "risk_level": "MODERATE"}
        ]
    },
    {
        "name": "Karen Thompson",
        "age": 44,
        "sex": "Female",
        "records": [
            {"resting_blood_pressure": 120, "cholesterol": 180, "fasting_blood_sugar": "No",
             "max_heart_rate_achieved": 170, "chest_pain_type": "Non-Anginal Pain",
             "rest_ecg": "Normal", "exercise_induced_angina": "No",
             "st_depression": 0.1, "st_slope": "Upsloping", "prediction": 0, "probability": 0.12, "risk_level": "LOW"}
        ]
    },
    {
        "name": "Joseph Garcia",
        "age": 63,
        "sex": "Male",
        "records": [
            {"resting_blood_pressure": 152, "cholesterol": 265, "fasting_blood_sugar": "Yes",
             "max_heart_rate_achieved": 130, "chest_pain_type": "Typical Angina",
             "rest_ecg": "Left Ventricular Hypertrophy", "exercise_induced_angina": "Yes",
             "st_depression": 3.2, "st_slope": "Downsloping", "prediction": 1, "probability": 0.95, "risk_level": "HIGH"}
        ]
    },
    {
        "name": "Lisa Martinez",
        "age": 49,
        "sex": "Female",
        "records": [
            {"resting_blood_pressure": 124, "cholesterol": 192, "fasting_blood_sugar": "No",
             "max_heart_rate_achieved": 163, "chest_pain_type": "Non-Anginal Pain",
             "rest_ecg": "Normal", "exercise_induced_angina": "No",
             "st_depression": 0.3, "st_slope": "Upsloping", "prediction": 0, "probability": 0.26, "risk_level": "LOW"}
        ]
    },
    {
        "name": "Daniel Robinson",
        "age": 56,
        "sex": "Male",
        "records": [
            {"resting_blood_pressure": 139, "cholesterol": 228, "fasting_blood_sugar": "No",
             "max_heart_rate_achieved": 143, "chest_pain_type": "Atypical Angina",
             "rest_ecg": "ST-T Wave Abnormality", "exercise_induced_angina": "No",
             "st_depression": 1.3, "st_slope": "Flat", "prediction": 1, "probability": 0.62, "risk_level": "MODERATE"}
        ]
    }
]

def add_sample_patients():
    """Add sample patients to the database"""
    db = PatientDatabase()
    
    print("Adding sample patients to database...")
    print("=" * 60)
    
    total_patients = 0
    total_records = 0
    
    for patient_data in sample_patients:
        # Add patient
        patient_id = db.add_patient(
            patient_data["name"],
            patient_data["age"],
            patient_data["sex"]
        )
        total_patients += 1
        print(f"[OK] Added patient: {patient_data['name']} (ID: {patient_id})")
        
        # Add records
        for record in patient_data["records"]:
            record_id = db.add_record(
                patient_id,
                record,
                record["prediction"],
                record["probability"],
                record["risk_level"]
            )
            total_records += 1
            print(f"  -> Added record {record_id} (Risk: {record['risk_level']}, Prob: {record['probability']:.2%})")
    
    print("=" * 60)
    print(f"\n[SUCCESS] Successfully added:")
    print(f"   - {total_patients} patients")
    print(f"   - {total_records} records")
    print(f"\nDatabase ready for use!")

if __name__ == "__main__":
    add_sample_patients()
