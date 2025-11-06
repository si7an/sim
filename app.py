from flask import Flask, render_template, request, jsonify, session
import os
from models.sepsis_model import SepsisPredictor
import pandas as pd
import json

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Initialize predictor
predictor = SepsisPredictor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        patient_data = [
            float(request.form['age']),
            float(request.form['heart_rate']),
            float(request.form['respiratory_rate']),
            float(request.form['temperature']),
            float(request.form['systolic_bp']),
            float(request.form['diastolic_bp']),
            float(request.form['wbc_count']),
            float(request.form['platelets']),
            float(request.form['creatinine']),
            float(request.form['glucose']),
            float(request.form['lactate']),
            float(request.form['sofa_score'])
        ]
        
        # Make prediction
        result = predictor.predict_sepsis(patient_data)
        
        # Store in session for history
        if 'history' not in session:
            session['history'] = []
        
        session['history'].append({
            'data': patient_data,
            'result': result
        })
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/train', methods=['POST'])
def train_model():
    try:
        accuracy = predictor.train_model()
        return jsonify({'message': f'Model trained successfully! Accuracy: {accuracy:.4f}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history')
def get_history():
    return jsonify(session.get('history', []))

@app.route('/clear_history', methods=['POST'])
def clear_history():
    session.pop('history', None)
    return jsonify({'message': 'History cleared'})

if __name__ == '__main__':
    # Ensure model is trained on first run
    if not os.path.exists('models/saved/sepsis_model.joblib'):
        print("Training model for first time use...")
        predictor.train_model()
    
    app.run(debug=True, host='0.0.0.0', port=5000)