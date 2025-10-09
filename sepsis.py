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

# Disease Severity Scores (Simplified for simulation, paper has SOFA, ASOFA, NEWS, MEWS etc.)
data.update({
    'SOFA_score': np.random.normal(4, 2, num_patients).clip(0, 15), # Capped to plausible range
})

# Biochemical Parameters & Biomarkers (Conceptual values, ranges based on paper's context)
data.update({
    'Albumin': np.random.normal(3.48, 0.54, num_patients).clip(1.5, 5.0), # g/dL
    'Platelet': np.random.normal(213.39, 96.82, num_patients).clip(50, 450), # 10^3 µL
    'Cortisol': np.random.normal(22.89, 16.56, num_patients).clip(5, 70), # µg/dL
    'Lactate': np.random.lognormal(np.log(2), 0.5, num_patients).clip(0.5, 10), # mg/dL, often log-normally distributed
    'Procalcitonin': np.random.lognormal(np.log(1), 1, num_patients).clip(0.01, 50), # ng/mL, often log-normally distributed
    'D_dimer': np.random.lognormal(np.log(1000), 0.8, num_patients).clip(100, 10000), # ng/mL, often log-normally distributed
    'IL_8': np.random.normal(50, 20, num_patients).clip(1, 200), # pg/mL
    'IL_6': np.random.normal(30, 15, num_patients).clip(1, 100), # pg/mL
    'Angiopoietin_2': np.random.normal(5, 2, num_patients).clip(0.5, 15), # ng/mL
    'Red_Blood_Cell': np.random.normal(4.17, 0.75, num_patients).clip(3.0, 6.0), # 10^6 µL
    'HCO3': np.random.normal(25.15, 4.19, num_patients).clip(15, 35), # mmol/L
    'AaDO2': np.random.normal(55.50, 30.36, num_patients).clip(10, 200), # mmHg
    'FDP': np.random.lognormal(np.log(18), 0.5, num_patients).clip(1, 100), # µg/mL
    'Uric_acid': np.random.normal(5.48, 2.30, num_patients).clip(2, 10), # mg/dL
    'SOFA_score_Res': np.random.normal(1, 0.5, num_patients).clip(0, 4), # Dummy for respiratory component of SOFA
    'E_selection': np.random.normal(40, 10, num_patients).clip(10, 80), # pg/mL
}

df = pd.DataFrame(data)

# Introduce some missing values to demonstrate imputation (e.g., 5% missing)
for col in ['SOFA_score', 'Albumin', 'Lactate', 'D_dimer', 'IL_8']:
    missing_indices = np.random.choice(df.index, size=int(num_patients * 0.05), replace=False)
    df.loc[missing_indices, col] = np.nan

# --- Target Variable: 28-day Mortality ---
# Create a simplified mortality outcome based on a combination of features.
# Higher SOFA, IL-8, D-dimer, Lactate, Malignancy, Age, and lower Albumin generally increase risk.
df['mortality_risk_score'] = (
    0.2 * df['SOFA_score'].fillna(df['SOFA_score'].median()) +
    0.05 * df['IL_8'].fillna(df['IL_8'].median()) +
    0.0005 * df['D_dimer'].fillna(df['D_dimer'].median()) +
    0.8 * (5 - df['Albumin'].fillna(df['Albumin'].median())) + # Lower albumin -> higher risk
    0.6 * df['Lactate'].fillna(df['Lactate'].median()) +
    0.1 * df['Malignancy'] +
    0.02 * df['Age']
)

# Set a threshold to achieve approximately 8% mortality
target_mortality_rate = 0.08
mortality_threshold = df['mortality_risk_score'].quantile(1 - target_mortality_rate)
df['28_day_mortality'] = (df['mortality_risk_score'] >= mortality_threshold).astype(int)

print(f"Synthetic dataset created with {len(df)} patients.")
print(f"Actual synthetic 28-day mortality rate: {df['28_day_mortality'].mean():.2%}\n")

# --- 2. Preprocessing Steps ---

# Identify continuous and categorical features
continuous_features = [
    'Age', 'Body_temperature', 'Pulse_rate', 'Respiratory_rate', 'SBP', 'DBP',
    'SOFA_score', 'Albumin', 'Platelet', 'Cortisol', 'Lactate', 'Procalcitonin',
    'D_dimer', 'IL_8', 'IL_6', 'Angiopoietin_2', 'Red_Blood_Cell', 'HCO3',
    'AaDO2', 'FDP', 'Uric_acid', 'SOFA_score_Res', 'E_selection'
]
categorical_features = ['Male_gender', 'Malignancy', 'Diabetes', 'Chronic_kidney_disease']

# Impute missing values
# For continuous features, use median imputation
for col in continuous_features:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())

# For categorical features, use mode imputation (less relevant if no missing in simulation)
for col in categorical_features:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode()[0])

print("Missing values imputed.")

# --- U-shaped feature transformation (conceptual example) ---
# The paper mentions centering and squaring for some U-shaped features.
# Let's apply this to 'Respiratory_rate' as an example, assuming an ideal rate.
# A moderate respiratory rate is healthy; too low or too high indicates distress.
ideal_resp_rate = df['Respiratory_rate'].mean() # Or a clinically determined ideal like 16-18
df['Respiratory_rate_U_shaped'] = (df['Respiratory_rate'] - ideal_resp_rate)**2
continuous_features.append('Respiratory_rate_U_shaped')
# Remove original respiratory rate from continuous features if we replace it
continuous_features.remove('Respiratory_rate')

print("U-shaped feature transformation applied for Respiratory_rate.")

# Feature selection (Paper used Boruta, here we define all relevant features for our synthetic model)
# We include all features generated and engineered, reflecting the "30 selected features" from the paper.
selected_features = continuous_features + categorical_features
X = df[selected_features]
y = df['28_day_mortality']

# Feature Scaling (StandardScaler recommended for many ML models)
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[continuous_features] = scaler.fit_transform(X[continuous_features])
print("Continuous features scaled.\n")

# --- 3. Split Data into Training and Testing Sets ---
# The paper used a 70/30 split, stratified by outcome to maintain class balance.
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y,
                                                    test_size=0.3,
                                                    random_state=42,
                                                    stratify=y)

print(f"Training set size: {len(X_train)} samples (Mortality: {y_train.mean():.2%})")
print(f"Testing set size: {len(X_test)} samples (Mortality: {y_test.mean():.2%})\n")

# --- 4. Model Training (Random Forest Classifier) ---
# Instantiate the Random Forest Classifier.
# The paper mentioned n_estimators=500 and max_features as '34' (implying a larger feature set).
# For our synthetic data, 'sqrt' is a good heuristic for max_features.
# 'class_weight="balanced"' helps address the imbalance between mortality (minority class) and survival.
rf_model = RandomForestClassifier(n_estimators=500,
                                  max_features='sqrt', # Common heuristic, or specify an int if needed
                                  random_state=42,
                                  class_weight='balanced',
                                  n_jobs=-1) # Use all available cores

print("Training Random Forest model...")
rf_model.fit(X_train, y_train)
print("Model training complete.\n")

# --- 5. Model Evaluation ---
# Predict probabilities for the positive class (mortality=1) on the test set
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

# Calculate AUROC
auroc = roc_auc_score(y_test, y_pred_proba)
print(f"Area Under Receiver Operating Characteristic Curve (AUROC) on test set: {auroc:.4f}")

# Generate a classification report using predicted classes (default threshold 0.5)
y_pred = rf_model.predict(X_test)
print("\nClassification Report (default 0.5 threshold):")
print(classification_report(y_test, y_pred))

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(10, 7))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUROC = {auroc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Receiver Operating Characteristic (ROC) Curve for 28-day Mortality Prediction')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
print("ROC Curve plot generated.")


# --- 6. Conceptual SHAP (Shapley Additive exPlanations) Summary Plot ---
# SHAP values help interpret individual predictions and overall feature importance.
# This part requires the 'shap' library (pip install shap).
try:
    import shap
    print("\nGenerating SHAP summary plot (this may take a moment)...")

    # Use TreeExplainer for tree-based models like Random Forest
    explainer = shap.TreeExplainer(rf_model)

    # Calculate SHAP values for the test set
    # shap_values for Random Forest typically return two arrays: one for class 0 (survival), one for class 1 (mortality)
    shap_values = explainer.shap_values(X_test)

    # The summary plot for binary classification usually focuses on the positive class (1 for mortality)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values[1], X_test, feature_names=selected_features, show=False)
    plt.title("SHAP Summary Plot for 28-day Mortality Prediction")
    plt.tight_layout()
    plt.show()
    print("SHAP Summary Plot generated. Higher SHAP value means higher impact towards predicting mortality (positive class).")

except ImportError:
    print("\nSHAP library not installed. Skipping SHAP plot generation.")
    print("To generate SHAP plots, please install it: pip install shap")
except Exception as e:
    print(f"\nAn error occurred during SHAP plot generation: {e}")


# --- 7. Demonstrate Prediction for a New, Unseen Patient ---
print("\n--- Demonstrating Prediction for a New Patient ---")

# Example data for a new hypothetical patient
# This patient shows several signs of higher risk (e.g., older, higher SOFA, low albumin, high lactate)
new_patient_raw_data = pd.DataFrame([{
    'Age': 75,
    'Male_gender': 1,
    'Malignancy': 1, # Has malignancy
    'Diabetes': 0,
    'Chronic_kidney_disease': 1, # Has CKD
    'Body_temperature': 37.0,
    'Pulse_rate': 120,
    'Respiratory_rate': 30, # High respiratory rate
    'SBP': 100,
    'DBP': 60,
    'SOFA_score': 10, # High SOFA
    'Albumin': 2.0, # Low albumin
    'Platelet': 80, # Low platelet
    'Cortisol': 18,
    'Lactate': 7.5, # High lactate
    'Procalcitonin': 25,
    'D_dimer': 8000, # High D-dimer
    'IL_8': 150, # High IL-8
    'IL_6': 80, # High IL-6
    'Angiopoietin_2': 12,
    'Red_Blood_Cell': 3.0,
    'HCO3': 18,
    'AaDO2': 180,
    'FDP': 70,
    'Uric_acid': 8,
    'SOFA_score_Res': 3,
    'E_selection': 70,
}])

# Apply the same preprocessing steps as the training data
# 1. U-shaped feature transformation
new_patient_raw_data['Respiratory_rate_U_shaped'] = (new_patient_raw_data['Respiratory_rate'] - ideal_resp_rate)**2
new_patient_processed = new_patient_raw_data[selected_features].copy()

# 2. Scale continuous features using the *same scaler* fitted on training data
new_patient_processed[continuous_features] = scaler.transform(new_patient_processed[continuous_features])

# Make prediction
prediction_proba_new_patient = rf_model.predict_proba(new_patient_processed)[:, 1]
prediction_class_new_patient = rf_model.predict(new_patient_processed)[0]

print(f"\nNew patient's predicted probability of 28-day mortality: {prediction_proba_new_patient[0]:.4f}")

if prediction_class_new_patient == 1:
    print("The model predicts this patient has a HIGH risk of 28-day mortality.")
else:
    print("The model predicts this patient has a LOW risk of 28-day mortality.")

print("\n--- End of Prediction Model Code ---")