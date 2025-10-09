import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Simulate Synthetic Dataset ---
# We'll create a dataset that conceptually reflects the features and target
# described in the paper. The actual relationships and distributions are
# simplified for demonstration purposes.
np.random.seed(42) # for reproducibility
num_patients = 1000

# Demographics & Comorbidities
data = {
    'Age': np.random.normal(62.48, 17.55, num_patients),
    'Male_gender': np.random.randint(0, 2, num_patients),
    'Malignancy': np.random.choice([0, 1], num_patients, p=[0.721, 0.279]), # ~28% malignancy from paper
    'Diabetes': np.random.choice([0, 1], num_patients, p=[0.604, 0.396]), # ~40% diabetes from paper
    'Chronic_kidney_disease': np.random.choice([0, 1], num_patients, p=[0.919, 0.081]), # ~8% CKD
}

# Vital Signs (Mean and SD based on Table 1)
data.update({
    'Body_temperature': np.random.normal(38.0, 1.26, num_patients),
    'Pulse_rate': np.random.normal(109, 21.42, num_patients),
    'Respiratory_rate': np.random.normal(21, 3.45, num_patients), # Used for U-shaped feature
    'SBP': np.random.normal(137, 30.33, num_patients),
    'DBP': np.random.normal(78, 17.34, num_patients),
})