"""
Utility functions for customer churn prediction.
Handles model loading, prediction, and explainable AI features.
"""

import os
import pandas as pd
import joblib
import warnings
import builtins
import numpy as np
from functools import lru_cache
from pathlib import Path

# Suppress numpy version mismatch warnings
warnings.filterwarnings("ignore", message=".*BitGenerator.*")
warnings.filterwarnings("ignore", message=".*numpy.random._mt19937.*")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ─────────────────────────────────────────────
#  Path Configuration
# ─────────────────────────────────────────────
# Get the directory of this file
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'model.pkl'
METADATA_PATH = BASE_DIR / 'metadata.pkl'

# Global variables for caching
_model = None
_metadata = None

# ─────────────────────────────────────────────
#  Model Loading with MT19937 Patch
# ─────────────────────────────────────────────
def get_model(force_reload=False):
    """
    Load the ML model with caching and MT19937 patching.
    
    Args:
        force_reload (bool): If True, reload model even if cached
        
    Returns:
        The loaded scikit-learn pipeline model
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        RuntimeError: If model loading fails
    """
    global _model
    
    # Return cached model if available
    if _model is not None and not force_reload:
        return _model
    
    # Check if model file exists
    if not MODEL_PATH.exists():
        error_msg = (
            f"Model file not found at {MODEL_PATH}. "
            "Please run: python modelapp/train_model.py"
        )
        print(f"ERROR: {error_msg}")
        raise FileNotFoundError(error_msg)
    
    # Patch for MT19937 BitGenerator mismatch between NumPy versions
    original_import = builtins.__import__
    
    def patched_import(name, *args, **kwargs):
        """Intercept imports of numpy.random modules that cause issues"""
        problem_modules = [
            'numpy.random._mt19937',
            'numpy.random.bit_generator', 
            'numpy.random.mt19937',
            'numpy.random._common'
        ]
        if name in problem_modules:
            return np.random
        return original_import(name, *args, **kwargs)
    
    try:
        # Apply the patch
        builtins.__import__ = patched_import
        
        # Load the model
        print(f"Loading model from {MODEL_PATH}...")
        _model = joblib.load(MODEL_PATH)
        print("✓ Model loaded successfully")
        
    except Exception as e:
        print(f"✗ Error loading model with joblib: {e}")
        
        # Fallback: try using pickle directly
        try:
            import pickle
            print("Attempting fallback with pickle...")
            with open(MODEL_PATH, 'rb') as f:
                _model = pickle.load(f)
            print("✓ Model loaded successfully with pickle fallback")
        except Exception as e2:
            error_msg = f"Failed to load model: {e} | Fallback error: {e2}"
            print(f"✗ {error_msg}")
            raise RuntimeError(error_msg)
    finally:
        # Always restore the original import function
        builtins.__import__ = original_import
    
    return _model


def get_metadata(force_reload=False):
    """
    Load metadata with caching.
    
    Args:
        force_reload (bool): If True, reload metadata even if cached
        
    Returns:
        dict: Metadata containing baseline values, feature lists, etc.
    """
    global _metadata
    
    if _metadata is not None and not force_reload:
        return _metadata
    
    if METADATA_PATH.exists():
        try:
            _metadata = joblib.load(METADATA_PATH)
            print(f"✓ Metadata loaded from {METADATA_PATH}")
        except Exception as e:
            print(f"✗ Error loading metadata: {e}")
            _metadata = _get_default_metadata()
    else:
        print(f"⚠ Metadata file not found at {METADATA_PATH}, using defaults")
        _metadata = _get_default_metadata()
    
    return _metadata


def _get_default_metadata():
    """Return default metadata structure if file not found"""
    return {
        'baseline': {},
        'numeric_features': [
            'Age', 'Subscription_Duration_Months', 'Monthly_Logins',
            'Last_Purchase_Days_Ago', 'App_Usage_Time_Min', 'Monthly_Spend',
            'Discount_Usage_Percentage', 'Customer_Support_Calls', 'Satisfaction_Score'
        ],
        'categorical_features': ['Contract_Type'],
        'model_performance': {},
        'training_date': None,
        'numpy_version': np.__version__
    }


# ─────────────────────────────────────────────
#  Risk Level Computation
# ─────────────────────────────────────────────
def compute_risk_level(prob: float) -> str:
    """
    Convert probability to risk level category.
    
    Args:
        prob (float): Churn probability between 0 and 1
        
    Returns:
        str: Risk level (Low, Medium, High, Critical)
    """
    if prob < 0.25:
        return "Low"
    elif prob < 0.50:
        return "Medium"
    elif prob < 0.75:
        return "High"
    else:
        return "Critical"


# ─────────────────────────────────────────────
#  Core Prediction Function
# ─────────────────────────────────────────────
def predict_for_customer(customer_data: dict):
    """
    Run a prediction for a single customer.
    
    Args:
        customer_data (dict): Dictionary with customer features
        
    Returns:
        dict: Prediction results with is_churn, probability, and risk_level
              or {'error': error_message} if failed
    """
    try:
        # Load model
        clf = get_model()
        
        # Convert to DataFrame (single row)
        df = pd.DataFrame([customer_data])
        
        # Make predictions
        prob = float(clf.predict_proba(df)[0][1])
        is_churn = bool(clf.predict(df)[0])
        risk_level = compute_risk_level(prob)
        
        return {
            'is_churn': is_churn,
            'churn_probability': prob,
            'risk_level': risk_level
        }
        
    except FileNotFoundError as e:
        print(f"Model file error: {e}")
        return {'error': 'Model not available. Please train the model first.'}
    except Exception as e:
        print(f"Error in predict_for_customer: {e}")
        return {'error': f'Prediction failed: {str(e)}'}


# ─────────────────────────────────────────────
#  Batch Prediction
# ─────────────────────────────────────────────
def predict_batch(customers_data: list):
    """
    Run predictions for multiple customers.
    
    Args:
        customers_data (list): List of customer data dictionaries
        
    Returns:
        list: List of prediction results
    """
    results = []
    for i, customer in enumerate(customers_data):
        result = predict_for_customer(customer)
        result['index'] = i
        if 'customer_id' in customer:
            result['customer_id'] = customer['customer_id']
        results.append(result)
    return results


# ─────────────────────────────────────────────
#  Explainable AI (XAI) Functions
# ─────────────────────────────────────────────
def get_risk_factors(data: dict, prob: float) -> list:
    """
    Identify key risk factors driving churn probability.
    
    Args:
        data (dict): Customer data
        prob (float): Churn probability
        
    Returns:
        list: Human-readable risk factors
    """
    # If probability is very low, customer is safe
    if prob < 0.1:
        return ["Customer profile is currently stable and healthy"]

    try:
        clf = get_model()
        meta = get_metadata()
    except Exception as e:
        print(f"Error loading model for risk factors: {e}")
        return ["Unable to analyze risk factors at this time"]
    
    baseline = meta.get('baseline', {})
    numeric_features = meta.get('numeric_features', [])
    categorical_features = meta.get('categorical_features', [])
    
    # If no baseline data, return generic message
    if not baseline:
        return ["Risk factors cannot be identified without baseline data"]
    
    impacts = []
    
    # 1. Sensitivity Analysis: How much would fixing each feature help?
    try:
        original_df = pd.DataFrame([data])
        base_prob = float(clf.predict_proba(original_df)[0][1])
    except Exception:
        base_prob = prob
    
    # Check numeric features
    for feat in numeric_features:
        if feat not in baseline or feat not in data:
            continue
        
        # Create a modified version with ideal value
        perturbed_data = data.copy()
        perturbed_data[feat] = baseline[feat]
        
        try:
            perturbed_df = pd.DataFrame([perturbed_data])
            new_prob = float(clf.predict_proba(perturbed_df)[0][1])
            reduction = base_prob - new_prob
            
            # If fixing this feature would reduce churn probability
            if reduction > 0.01:  # At least 1% reduction
                impacts.append((feat, reduction))
        except Exception:
            continue
    
    # Check categorical features
    for feat in categorical_features:
        if feat not in baseline or feat not in data:
            continue
        
        if data[feat] != baseline[feat]:
            impacts.append((feat, 0.1))  # Fixed weight for categorical mismatch
    
    # Sort by impact (highest first)
    impacts.sort(key=lambda x: x[1], reverse=True)
    
    # 2. Fallback: If sensitivity analysis didn't work, use deviation analysis
    if not impacts:
        for feat in numeric_features:
            if feat not in baseline or feat not in data:
                continue
            
            val = data[feat]
            base = baseline.get(feat, val)
            if base == 0:
                base = 1
            
            # Determine if higher or lower is better
            if feat in ['Customer_Support_Calls', 'Last_Purchase_Days_Ago', 'Discount_Usage_Percentage']:
                # Lower is better
                if val > base:
                    deviation = (val - base) / base
                    if deviation > 0.2:  # More than 20% worse
                        impacts.append((feat, deviation))
            else:
                # Higher is better
                if val < base:
                    deviation = (base - val) / base
                    if deviation > 0.2:  # More than 20% worse
                        impacts.append((feat, deviation))
        
        impacts.sort(key=lambda x: x[1], reverse=True)

    # Map feature names to human-readable reasons
    reason_map = {
        'Satisfaction_Score': "Low satisfaction score",
        'Customer_Support_Calls': "Frequent support calls",
        'Last_Purchase_Days_Ago': "Long inactivity period",
        'Monthly_Logins': "Infrequent platform usage",
        'Monthly_Spend': "Below average spending",
        'Contract_Type': f"High-risk contract type ({data.get('Contract_Type', 'unknown')})",
        'App_Usage_Time_Min': "Decreasing usage time",
        'Discount_Usage_Percentage': "High discount dependency",
        'Subscription_Duration_Months': "Short subscription history",
        'Age': "Age-related risk pattern"
    }
    
    # Take top 3 factors
    factors = []
    for feat_name, _ in impacts[:3]:
        label = reason_map.get(feat_name, feat_name.replace('_', ' '))
        factors.append(label)
    
    return factors if factors else ["Complex combination of factors"]


def get_recommendations(is_churn: bool, prob: float, data: dict) -> list:
    """
    Generate actionable recommendations based on risk factors.
    
    Args:
        is_churn (bool): Whether customer is predicted to churn
        prob (float): Churn probability
        data (dict): Customer data
        
    Returns:
        list: Actionable recommendations
    """
    # For very safe customers
    if not is_churn and prob < 0.2:
        return [
            "Maintain engagement with loyalty rewards",
            "Send regular product updates and tips",
            "Consider enrolling in referral program"
        ]

    # Get risk factors
    factors = get_risk_factors(data, prob)
    
    # Recommendation mapping
    recommendation_map = {
        "satisfaction": [
            "Send satisfaction survey with incentive",
            "Schedule a feedback call with customer success",
            "Address any negative feedback immediately"
        ],
        "support": [
            "Review support ticket history",
            "Assign dedicated support representative",
            "Proactive check-in to resolve pending issues"
        ],
        "inactivity": [
            "Send personalized re-engagement email",
            "Offer limited-time return incentive",
            "Share new features they haven't tried"
        ],
        "usage": [
            "Provide usage tips and best practices",
            "Offer free training session",
            "Highlight underutilized features"
        ],
        "spending": [
            "Review pricing plan suitability",
            "Offer bundle discount or upgrade incentive",
            "Create personalized value proposition"
        ],
        "contract": [
            "Discuss benefits of longer-term contract",
            "Offer discount for annual commitment",
            "Highlight stability and savings"
        ],
        "discount": [
            "Review discount dependency",
            "Create value-based retention offer",
            "Transition to sustainable pricing"
        ],
        "subscription": [
            "Welcome and nurture new customer",
            "Accelerate time-to-value",
            "Provide onboarding success checklist"
        ]
    }
    
    recommendations = []
    
    # Match factors to recommendations
    for factor in factors:
        factor_lower = factor.lower()
        matched = False
        
        for key, recs in recommendation_map.items():
            if key in factor_lower:
                recommendations.extend(recs)
                matched = True
                break
        
        if not matched:
            recommendations.append(f"Investigate: {factor}")
    
    # Add universal recommendations based on risk level
    if prob > 0.7:
        recommendations.append("Escalate to retention team for immediate intervention")
    elif prob > 0.5:
        recommendations.append("Schedule account review within 7 days")
    
    # Remove duplicates and limit to top 5
    unique_recs = []
    for rec in recommendations:
        if rec not in unique_recs:
            unique_recs.append(rec)
    
    return unique_recs[:5]


# ─────────────────────────────────────────────
#  Initialize on module load (optional)
# ─────────────────────────────────────────────
def initialize():
    """Pre-load model and metadata to catch issues early"""
    try:
        get_model()
        get_metadata()
        print("✓ Model and metadata initialized successfully")
        return True
    except Exception as e:
        print(f"⚠ Warning: Could not pre-load model: {e}")
        return False

# Uncomment to pre-load when module is imported
# initialize()