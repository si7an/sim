import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

class SepsisPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'age', 'heart_rate', 'respiratory_rate', 'temperature', 
            'systolic_bp', 'diastolic_bp', 'wbc_count', 'platelets',
            'creatinine', 'glucose', 'lactate', 'sofa_score'
        ]
        
    def generate_sample_data(self, n_samples=1000):
        """Generate synthetic sepsis data for demonstration"""
        np.random.seed(42)
        
        data = {
            'age': np.random.randint(18, 90, n_samples),
            'heart_rate': np.random.randint(60, 140, n_samples),
            'respiratory_rate': np.random.randint(12, 35, n_samples),
            'temperature': np.random.normal(37, 1.5, n_samples),
            'systolic_bp': np.random.randint(80, 200, n_samples),
            'diastolic_bp': np.random.randint(50, 120, n_samples),
            'wbc_count': np.random.normal(8, 4, n_samples),
            'platelets': np.random.normal(250, 100, n_samples),
            'creatinine': np.random.normal(1.0, 0.5, n_samples),
            'glucose': np.random.normal(120, 40, n_samples),
            'lactate': np.random.normal(1.5, 1.0, n_samples),
            'sofa_score': np.random.randint(0, 15, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Create target variable based on realistic patterns
        sepsis_risk = (
            (df['age'] > 65) * 0.3 +
            (df['heart_rate'] > 100) * 0.2 +
            (df['respiratory_rate'] > 22) * 0.2 +
            (df['temperature'] > 38.3) * 0.1 +
            (df['wbc_count'] > 12) * 0.3 +
            (df['platelets'] < 150) * 0.2 +
            (df['creatinine'] > 1.2) * 0.2 +
            (df['lactate'] > 2.0) * 0.4 +
            (df['sofa_score'] > 6) * 0.5
        )
        
        df['sepsis_positive'] = (sepsis_risk > 0.7).astype(int)
        return df
    
    def train_model(self):
        """Train the sepsis prediction model"""
        # Generate or load your dataset
        df = self.generate_sample_data(2000)
        
        # Prepare features and target
        X = df[self.feature_names]
        y = df['sepsis_positive']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model trained with accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))
        
        # Save model and scaler
        self.save_model()
        
        return accuracy
    
    def predict_sepsis(self, patient_data):
        """Predict sepsis for a single patient"""
        if self.model is None:
            self.load_model()
        
        # Convert to DataFrame
        patient_df = pd.DataFrame([patient_data], columns=self.feature_names)
        
        # Scale features
        patient_scaled = self.scaler.transform(patient_df)
        
        # Make prediction
        prediction = self.model.predict(patient_scaled)[0]
        probability = self.model.predict_proba(patient_scaled)[0][1]
        
        return {
            'prediction': 'Positive' if prediction == 1 else 'Negative',
            'probability': round(probability * 100, 2),
            'risk_level': self.get_risk_level(probability)
        }
    
    def get_risk_level(self, probability):
        """Determine risk level based on probability"""
        if probability < 0.3:
            return "Low Risk"
        elif probability < 0.7:
            return "Medium Risk"
        else:
            return "High Risk"
    
    def save_model(self):
        """Save trained model and scaler"""
        os.makedirs('models/saved', exist_ok=True)
        joblib.dump(self.model, 'models/saved/sepsis_model.joblib')
        joblib.dump(self.scaler, 'models/saved/scaler.joblib')
    
    def load_model(self):
        """Load trained model and scaler"""
        try:
            self.model = joblib.load('models/saved/sepsis_model.joblib')
            self.scaler = joblib.load('models/saved/scaler.joblib')
        except:
            print("Model not found. Training new model...")
            self.train_model()