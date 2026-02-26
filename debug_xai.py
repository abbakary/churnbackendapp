import os
import joblib
import pandas as pd

# Use absolute paths to be sure
BASE_DIR = r"C:\New folder\new\customerPrediction\customerPrediction\customerchurn\modelapp"
METADATA_PATH = os.path.join(BASE_DIR, 'metadata.pkl')
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')

print(f"Checking Metadata at: {METADATA_PATH}")
if os.path.exists(METADATA_PATH):
    meta = joblib.load(METADATA_PATH)
    print("\n--- BASELINE (Safe Profile) ---")
    import json
    print(json.dumps(meta.get('baseline', {}), indent=2))
    
    print("\n--- FEATURES ---")
    print(f"Numeric: {meta.get('numeric_features')}")
    print(f"Categorical: {meta.get('categorical_features')}")
else:
    print("Metadata file NOT FOUND!")

print(f"\nChecking Model at: {MODEL_PATH}")
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
    
    # Test a "high risk" dummy customer
    test_data = {
        'Age': 45,
        'Subscription_Duration_Months': 5,
        'Contract_Type': 'Monthly',
        'Monthly_Logins': 2,
        'Last_Purchase_Days_Ago': 65,
        'App_Usage_Time_Min': 5,
        'Monthly_Spend': 20,
        'Discount_Usage_Percentage': 50,
        'Customer_Support_Calls': 8,
        'Satisfaction_Score': 1
    }
    df = pd.DataFrame([test_data])
    prob = model.predict_proba(df)[0][1]
    print(f"\nTest Customer Churn Prob: {prob:.4f}")
    
    # Try perturbation
    baseline = meta.get('baseline', {})
    for feat in meta.get('numeric_features', []) + meta.get('categorical_features', []):
        perturbed = test_data.copy()
        perturbed[feat] = baseline.get(feat, test_data[feat])
        p_df = pd.DataFrame([perturbed])
        new_prob = model.predict_proba(p_df)[0][1]
        reduction = prob - new_prob
        print(f"Feat: {feat:30} | New Prob: {new_prob:.4f} | Reduction: {reduction:.4f}")

else:
    print("Model file NOT FOUND!")
