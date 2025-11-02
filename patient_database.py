"""
Patient Database Management
Handles storage and retrieval of patient records
"""

import sqlite3
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

class PatientDatabase:
    def __init__(self, db_path: str = 'patients.db'):
        """Initialize the patient database"""
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Patients table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patients (
                patient_id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_name TEXT NOT NULL,
                age INTEGER NOT NULL,
                sex TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Patient records table (for multiple records per patient)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patient_records (
                record_id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                resting_blood_pressure INTEGER,
                cholesterol INTEGER,
                fasting_blood_sugar TEXT,
                max_heart_rate_achieved INTEGER,
                chest_pain_type TEXT,
                rest_ecg TEXT,
                exercise_induced_angina TEXT,
                st_depression REAL,
                st_slope TEXT,
                prediction INTEGER,
                probability REAL,
                risk_level TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
            )
        ''')
        
        # Reports table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reports (
                report_id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER,
                record_id INTEGER,
                report_text TEXT,
                report_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (patient_id) REFERENCES patients (patient_id),
                FOREIGN KEY (record_id) REFERENCES patient_records (record_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_patient(self, patient_name: str, age: int, sex: str) -> int:
        """Add a new patient and return patient_id"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO patients (patient_name, age, sex)
            VALUES (?, ?, ?)
        ''', (patient_name, age, sex))
        
        patient_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return patient_id
    
    def add_record(self, patient_id: int, patient_data: Dict, prediction: int, 
                   probability: float, risk_level: str) -> int:
        """Add a patient record and return record_id"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO patient_records (
                patient_id, resting_blood_pressure, cholesterol, fasting_blood_sugar,
                max_heart_rate_achieved, chest_pain_type, rest_ecg,
                exercise_induced_angina, st_depression, st_slope,
                prediction, probability, risk_level
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            patient_id,
            patient_data.get('resting_blood_pressure'),
            patient_data.get('cholesterol'),
            patient_data.get('fasting_blood_sugar'),
            patient_data.get('max_heart_rate_achieved'),
            patient_data.get('chest_pain_type'),
            patient_data.get('rest_ecg'),
            patient_data.get('exercise_induced_angina'),
            patient_data.get('st_depression'),
            patient_data.get('st_slope'),
            prediction,
            probability,
            risk_level
        ))
        
        record_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return record_id
    
    def add_report(self, patient_id: int, record_id: int, report_text: str, 
                   report_json: Dict) -> int:
        """Add a report and return report_id"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO reports (patient_id, record_id, report_text, report_json)
            VALUES (?, ?, ?, ?)
        ''', (patient_id, record_id, report_text, json.dumps(report_json)))
        
        report_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return report_id
    
    def get_patient(self, patient_id: int) -> Optional[Dict]:
        """Get patient information by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT patient_id, patient_name, age, sex, created_at, updated_at
            FROM patients WHERE patient_id = ?
        ''', (patient_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'patient_id': row[0],
                'patient_name': row[1],
                'age': row[2],
                'sex': row[3],
                'created_at': row[4],
                'updated_at': row[5]
            }
        return None
    
    def get_patient_records(self, patient_id: int) -> List[Dict]:
        """Get all records for a patient"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query('''
            SELECT * FROM patient_records
            WHERE patient_id = ?
            ORDER BY created_at DESC
        ''', conn, params=(patient_id,))
        conn.close()
        return df.to_dict('records')
    
    def get_all_patients(self) -> pd.DataFrame:
        """Get all patients"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query('''
            SELECT p.*, 
                   COUNT(pr.record_id) as total_records,
                   MAX(pr.created_at) as last_record_date
            FROM patients p
            LEFT JOIN patient_records pr ON p.patient_id = pr.patient_id
            GROUP BY p.patient_id
            ORDER BY p.created_at DESC
        ''', conn)
        conn.close()
        return df
    
    def get_record_with_patient(self, record_id: int) -> Optional[Dict]:
        """Get a record with patient information"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT pr.*, p.patient_name, p.age, p.sex
            FROM patient_records pr
            JOIN patients p ON pr.patient_id = p.patient_id
            WHERE pr.record_id = ?
        ''', (record_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, row))
        return None
    
    def get_reports_for_patient(self, patient_id: int) -> List[Dict]:
        """Get all reports for a patient"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query('''
            SELECT * FROM reports
            WHERE patient_id = ?
            ORDER BY created_at DESC
        ''', conn, params=(patient_id,))
        conn.close()
        return df.to_dict('records')
    
    def get_report_for_record(self, record_id: int) -> Optional[Dict]:
        """Get report for a specific record"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM reports
            WHERE record_id = ?
        ''', (record_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, row))
        return None
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total patients
        cursor.execute('SELECT COUNT(*) FROM patients')
        total_patients = cursor.fetchone()[0]
        
        # Total records
        cursor.execute('SELECT COUNT(*) FROM patient_records')
        total_records = cursor.fetchone()[0]
        
        # Risk level distribution
        cursor.execute('''
            SELECT risk_level, COUNT(*) as count
            FROM patient_records
            GROUP BY risk_level
        ''')
        risk_distribution = dict(cursor.fetchall())
        
        # Prediction distribution
        cursor.execute('''
            SELECT prediction, COUNT(*) as count
            FROM patient_records
            GROUP BY prediction
        ''')
        prediction_distribution = dict(cursor.fetchall())
        
        # Average probability by risk level
        cursor.execute('''
            SELECT risk_level, AVG(probability) as avg_prob
            FROM patient_records
            GROUP BY risk_level
        ''')
        avg_probabilities = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            'total_patients': total_patients,
            'total_records': total_records,
            'risk_distribution': risk_distribution,
            'prediction_distribution': prediction_distribution,
            'avg_probabilities': avg_probabilities
        }
    
    def search_patients(self, search_term: str) -> pd.DataFrame:
        """Search patients by name"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query('''
            SELECT p.*, 
                   COUNT(pr.record_id) as total_records,
                   MAX(pr.created_at) as last_record_date
            FROM patients p
            LEFT JOIN patient_records pr ON p.patient_id = pr.patient_id
            WHERE p.patient_name LIKE ?
            GROUP BY p.patient_id
            ORDER BY p.created_at DESC
        ''', conn, params=(f'%{search_term}%',))
        conn.close()
        return df
    
    def delete_patient(self, patient_id: int):
        """Delete a patient and all related records"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Delete reports
        cursor.execute('DELETE FROM reports WHERE patient_id = ?', (patient_id,))
        
        # Delete records
        cursor.execute('DELETE FROM patient_records WHERE patient_id = ?', (patient_id,))
        
        # Delete patient
        cursor.execute('DELETE FROM patients WHERE patient_id = ?', (patient_id,))
        
        conn.commit()
        conn.close()
    
    def update_patient(self, patient_id: int, patient_name: str = None, 
                      age: int = None, sex: str = None):
        """Update patient information"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        updates = []
        params = []
        
        if patient_name:
            updates.append('patient_name = ?')
            params.append(patient_name)
        if age:
            updates.append('age = ?')
            params.append(age)
        if sex:
            updates.append('sex = ?')
            params.append(sex)
        
        if updates:
            updates.append('updated_at = CURRENT_TIMESTAMP')
            params.append(patient_id)
            
            cursor.execute(f'''
                UPDATE patients
                SET {', '.join(updates)}
                WHERE patient_id = ?
            ''', params)
            
            conn.commit()
        
        conn.close()
