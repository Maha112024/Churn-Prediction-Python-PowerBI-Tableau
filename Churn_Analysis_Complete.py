# ============================================================
# CUSTOMER CHURN PREDICTION — COMPLETE PROJECT
# Decision Science Analysis
# Author: Maha Pentakota
# Date: May 2026
# ============================================================

# ============================================================
# STEP 1: IMPORT LIBRARIES
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("=" * 60)
print("🚀 CUSTOMER CHURN PREDICTION PROJECT")
print("=" * 60)

# ============================================================
# STEP 2: LOAD DATA
# ============================================================
df = pd.read_csv('Telco-Customer-Churn.csv')

print("\n📁 DATA LOADED!")
print(f"   Rows: {df.shape[0]}")
print(f"   Columns: {df.shape[1]}")

# ============================================================
# STEP 3: EXPLORE DATA
# ============================================================
print("\n" + "=" * 60)
print("👀 FIRST 5 ROWS:")
print("=" * 60)
print(df.head())

print("\n📋 DATA TYPES:")
print(df.dtypes)

# ============================================================
# STEP 4: CHECK FOR ISSUES
# ============================================================
print("\n" + "=" * 60)
print("🔍 CHECKING FOR DATA ISSUES:")
print("=" * 60)

# Check nulls
print(f"   Null values: {df.isnull().sum().sum()}")

# Check blanks in text columns
for col in df.select_dtypes(include=['object']).columns:
    blanks = (df[col].str.strip() == '').sum()
    if blanks > 0:
        print(f"   ⚠️ {col}: {blanks} blanks found")

# Check duplicates
print(f"   Duplicate rows: {df.duplicated().sum()}")

# ============================================================
# STEP 5: CLEAN DATA
# ============================================================
print("\n" + "=" * 60)
print("🧹 CLEANING DATA:")
print("=" * 60)

# Fix TotalCharges (convert to number, fill blanks with median)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
print(f"   ✅ Fixed TotalCharges (median: ${df['TotalCharges'].median():.2f})")

# Drop customerID (not useful for analysis)
df = df.drop('customerID', axis=1)
print(f"   ✅ Dropped customerID")

# Create binary target variable
df['Churn_Binary'] = df['Churn'].map({'Yes': 1, 'No': 0})
print(f"   ✅ Created Churn_Binary column")

print(f"\n   Final shape: {df.shape[0]} rows, {df.shape[1]} columns")

# ============================================================
# STEP 6: CHURN DISTRIBUTION
# ============================================================
print("\n" + "=" * 60)
print("📊 CHURN DISTRIBUTION:")
print("=" * 60)
churn_counts = df['Churn'].value_counts()
churn_pct = round(df['Churn'].value_counts(normalize=True) * 100, 1)
print(f"   No (Stayed):  {churn_counts['No']} ({churn_pct['No']}%)")
print(f"   Yes (Left):   {churn_counts['Yes']} ({churn_pct['Yes']}%)")

# ============================================================
# STEP 7: SAVE CLEAN DATA
# ============================================================
print("\n" + "=" * 60)
print("💾 SAVING CLEAN DATA:")
print("=" * 60)

# Save to same folder
save_path = 'Telco_Churn_Clean.csv'
df.to_csv(save_path, index=False)
print(f"   ✅ Saved: {os.path.abspath(save_path)}")

# ============================================================
# STEP 8: EDA — VISUALIZATIONS
# ============================================================
print("\n" + "=" * 60)
print("📈 CREATING VISUALIZATIONS:")
print("=" * 60)

# Chart 1: Churn Pie Chart
plt.figure(figsize=(6, 6))
df['Churn'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['#66b3ff', '#ff6666'])
plt.title('Customer Churn Distribution')
plt.ylabel('')
plt.savefig('01_churn_pie.png')
plt.close()
print("   ✅ Saved: 01_churn_pie.png")

# Chart 2: Churn by Contract Type
plt.figure(figsize=(8, 5))
sns.countplot(x='Contract', hue='Churn', data=df)
plt.title('Churn by Contract Type')
plt.savefig('02_churn_by_contract.png')
plt.close()
print("   ✅ Saved: 02_churn_by_contract.png")

# Chart 3: Tenure by Churn
plt.figure(figsize=(8, 5))
sns.boxplot(x='Churn', y='tenure', data=df)
plt.title('Tenure by Churn Status')
plt.savefig('03_tenure_by_churn.png')
plt.close()
print("   ✅ Saved: 03_tenure_by_churn.png")

# Chart 4: Monthly Charges by Churn
plt.figure(figsize=(8, 5))
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title('Monthly Charges by Churn Status')
plt.savefig('04_charges_by_churn.png')
plt.close()
print("   ✅ Saved: 04_charges_by_churn.png")

# Chart 5: Churn by Payment Method
plt.figure(figsize=(10, 5))
sns.countplot(x='PaymentMethod', hue='Churn', data=df)
plt.title('Churn by Payment Method')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('05_churn_by_payment.png')
plt.close()
print("   ✅ Saved: 05_churn_by_payment.png")

# ============================================================
# STEP 9: BUILD PREDICTION MODEL
# ============================================================
print("\n" + "=" * 60)
print("🤖 BUILDING PREDICTION MODEL:")
print("=" * 60)

# Select features
features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
X = df[features]
y = df['Churn_Binary']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"   Training: {X_train.shape[0]} rows")
print(f"   Testing:  {X_test.shape[0]} rows")

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Results
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"\n   🎯 ACCURACY: {accuracy:.1f}%")

# Feature Importance
print("\n   📋 FEATURE IMPORTANCE (What Drives Churn):")
importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

for _, row in importance.iterrows():
    print(f"      • {row['Feature']}: {row['Importance']*100:.1f}%")

# ============================================================
# STEP 10: KEY INSIGHTS
# ============================================================
print("\n" + "=" * 60)
print("💡 KEY INSIGHTS:")
print("=" * 60)
print("   1. Churn Rate: 26.5% of customers left")
print("   2. Month-to-month contracts have highest churn")
print("   3. Higher Monthly Charges = More likely to churn")
print("   4. New customers (low tenure) churn more")
print("   5. Electronic check users churn most")

# ============================================================
# COMPLETE!
# ============================================================
print("\n" + "=" * 60)
print("🎉 PROJECT COMPLETE!")
print("=" * 60)
print("\n📁 FILES CREATED:")
print(f"   • Telco_Churn_Clean.csv (for Power BI & Tableau)")
print(f"   • 01_churn_pie.png")
print(f"   • 02_churn_by_contract.png")
print(f"   • 03_tenure_by_churn.png")
print(f"   • 04_charges_by_churn.png")
print(f"   • 05_churn_by_payment.png")
print(f"\n📍 Location: {os.getcwd()}")
print("\n✅ Ready for Power BI & Tableau!")
