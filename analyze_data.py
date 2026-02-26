import pandas as pd
import json

CSV_PATH = r"C:\New folder\new\customerPrediction\customerPrediction\customerchurn\modelapp\model\churn.csv"
df = pd.read_csv(CSV_PATH)

print("--- Contract_Type vs Churn ---")
print(pd.crosstab(df['Contract_Type'], df['Is_Churn']))

print("\n--- Satisfaction_Score vs Churn (Means) ---")
print(df.groupby('Is_Churn')['Satisfaction_Score'].mean())

print("\n--- Customer_Support_Calls vs Churn (Means) ---")
print(df.groupby('Is_Churn')['Customer_Support_Calls'].mean())

# Find the "ideal" safe value (e.g. 90th percentile of safe population or 10th depending on direction)
safe_df = df[df['Is_Churn'] == 0]
ideal_baseline = {}

# For features where lower is better (Churn, Support calls, Last purchase days)
for col in ['Customer_Support_Calls', 'Last_Purchase_Days_Ago', 'Discount_Usage_Percentage']:
    ideal_baseline[col] = safe_df[col].quantile(0.1) # 10th percentile

# For features where higher is better (Satisfaction, App Usage, Logins, Spend, Duration)
for col in ['Satisfaction_Score', 'App_Usage_Time_Min', 'Monthly_Logins', 'Monthly_Spend', 'Subscription_Duration_Months']:
    ideal_baseline[col] = safe_df[col].quantile(0.9) # 90th percentile

# For Contract_Type, we want the most "stable" one
# Let's see which one has the lowest churn rate
churn_rates = df.groupby('Contract_Type')['Is_Churn'].mean()
ideal_baseline['Contract_Type'] = churn_rates.idxmin()

print("\n--- Ideal Baseline (Top 10% performance) ---")
print(json.dumps(ideal_baseline, indent=2))
