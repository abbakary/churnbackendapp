import os
import sys

# Add current directory to path
sys.path.append(os.getcwd())

# Mock Django settings if needed, but get_model in utils.py seems self-contained regarding paths
# Actually, utils.py imports pandas, joblib, builtins, numpy, warnings
# It calculates MODEL_PATH based on __file__

try:
    from modelapp.utils import get_model
    print("Attempting to load model...")
    model = get_model()
    print("Model loaded successfully!")
    
    # Test a prediction
    import pandas as pd
    test_data = {
        'Age': 30,
        'Subscription_Duration_Months': 12,
        'Contract_Type': 'Monthly',
        'Monthly_Logins': 10,
        'Last_Purchase_Days_Ago': 5,
        'App_Usage_Time_Min': 100.0,
        'Monthly_Spend': 50.0,
        'Discount_Usage_Percentage': 10.0,
        'Customer_Support_Calls': 1,
        'Satisfaction_Score': 4
    }
    df = pd.DataFrame([test_data])
    prob = model.predict_proba(df)[0][1]
    print(f"Test prediction successful! Churn probability: {prob}")
    
except Exception as e:
    print(f"VERIFICATION FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
