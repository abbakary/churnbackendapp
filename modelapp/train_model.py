"""
Train and save the Customer Churn GradientBoosting model.

Run this script once from the `customerchurn` Django project root:
    python modelapp/train_model.py

It will:
  1. Load the real churn dataset (churn.csv)
  2. Train a GradientBoostingClassifier pipeline
  3. Save model.pkl and metadata.pkl to the same directory as this file
"""

import os
import pandas as pd
import joblib
import warnings
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# Suppress warnings during training
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Updated path to match your project structure
CSV_PATH = os.path.join(SCRIPT_DIR, 'model', 'churn.csv')
MODEL_PATH = os.path.join(SCRIPT_DIR, 'model.pkl')
METADATA_PATH = os.path.join(SCRIPT_DIR, 'metadata.pkl')

# ─── Train Model ──────────────────────────────────────────────────────────────
def train():
    print("Loading real churn dataset...")
    
    # Check if CSV exists
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: CSV file not found at {CSV_PATH}")
        print("Please update the CSV_PATH variable to point to your churn.csv file")
        return None
    
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded dataset → {CSV_PATH}")
    print(f"Dataset shape: {df.shape}")
    print(f"Dataset columns: {df.columns.tolist()}")
    print(f"Target distribution:\n{df['Is_Churn'].value_counts()}")

    # Target column is 'Is_Churn'
    X = df.drop(['Is_Churn', 'CustomerID'], axis=1)
    y = df['Is_Churn']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Define numeric and categorical features based on your dataset head
    numeric_features = [
        'Age',
        'Subscription_Duration_Months',
        'Monthly_Logins',
        'Last_Purchase_Days_Ago',
        'App_Usage_Time_Min',
        'Monthly_Spend',
        'Discount_Usage_Percentage',
        'Customer_Support_Calls',
        'Satisfaction_Score'
    ]
    categorical_features = ['Contract_Type']

    # Verify all features exist in the dataset
    available_numeric = [f for f in numeric_features if f in X.columns]
    available_categorical = [f for f in categorical_features if f in X.columns]
    
    missing_numeric = set(numeric_features) - set(available_numeric)
    missing_categorical = set(categorical_features) - set(available_categorical)
    
    if missing_numeric:
        print(f"Warning: Missing numeric features: {missing_numeric}")
    if missing_categorical:
        print(f"Warning: Missing categorical features: {missing_categorical}")
    
    if not available_numeric and not available_categorical:
        print("ERROR: No features available for training!")
        return None

    # Build preprocessor with available features
    transformers = []
    if available_numeric:
        transformers.append(('num', StandardScaler(), available_numeric))
    if available_categorical:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), available_categorical))
    
    preprocessor = ColumnTransformer(transformers=transformers)

    # Create pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=4,
            random_state=42,
            min_samples_split=2,
            min_samples_leaf=1,
            subsample=1.0,
            max_features='sqrt',
            verbose=0
        )),
    ])

    print("Training GradientBoostingClassifier...")
    print(f"Using numeric features: {available_numeric}")
    print(f"Using categorical features: {available_categorical}")
    
    pipeline.fit(X_train, y_train)

    # Make predictions
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    print(f"\n=== Model Performance ===")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")

    # ─── Calculate Metadata for Reasoning ───────────────────────────────────
    # Calculate "Ideal Safe Baseline": Values that strongly signal "Safe" (Non-churn)
    safe_df = df[df['Is_Churn'] == 0]
    baseline = {}
    
    # Lower is better features
    lower_better = ['Customer_Support_Calls', 'Last_Purchase_Days_Ago', 'Discount_Usage_Percentage']
    for col in lower_better:
        if col in df.columns:
            baseline[col] = float(safe_df[col].quantile(0.1))
            
    # Higher is better features
    higher_better = ['Satisfaction_Score', 'App_Usage_Time_Min', 'Monthly_Logins', 
                     'Monthly_Spend', 'Subscription_Duration_Months', 'Age']
    for col in higher_better:
        if col in df.columns:
            baseline[col] = float(safe_df[col].quantile(0.9))

    # Contract Type: lowest churn rate
    if 'Contract_Type' in df.columns:
        churn_rates = df.groupby('Contract_Type')['Is_Churn'].mean()
        baseline['Contract_Type'] = churn_rates.idxmin()

    metadata = {
        'baseline': baseline,
        'numeric_features': available_numeric,
        'categorical_features': available_categorical,
        'model_performance': {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob)
        },
        'training_date': pd.Timestamp.now().isoformat(),
        'numpy_version': np.__version__,
        'sklearn_version': sklearn.__version__
    }

    # Save model and metadata
    joblib.dump(pipeline, MODEL_PATH)
    joblib.dump(metadata, METADATA_PATH)

    print(f"\nModel saved → {MODEL_PATH}")
    print(f"Metadata saved → {METADATA_PATH}")
    
    # Verify the model loads correctly
    try:
        test_load = joblib.load(MODEL_PATH)
        print("✓ Model verification: Successfully loaded")
    except Exception as e:
        print(f"✗ Model verification failed: {e}")
    
    return pipeline

def load_and_test_model():
    """Test loading the saved model"""
    print("\n=== Testing Model Loading ===")
    try:
        model = joblib.load(MODEL_PATH)
        metadata = joblib.load(METADATA_PATH)
        print("✓ Model and metadata loaded successfully")
        print(f"Model type: {type(model)}")
        print(f"Metadata contains: {list(metadata.keys())}")
        return model, metadata
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return None, None

if __name__ == '__main__':
    # Add sklearn import here to avoid circular imports
    import sklearn
    print(f"NumPy version: {np.__version__}")
    print(f"scikit-learn version: {sklearn.__version__}")
    print(f"Joblib version: {joblib.__version__}")
    
    # Train the model
    model = train()
    
    # Test loading
    if model:
        load_and_test_model()