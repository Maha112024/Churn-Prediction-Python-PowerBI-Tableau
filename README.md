# Churn-Prediction-Python-PowerBI-Tableau
# Customer Churn Prediction

## 📊 Project Overview
Analyzed customer churn patterns for a telecom company and built a machine learning model to predict which customers are likely to leave. Created interactive dashboards in both Power BI and Tableau.

## 🎯 Key Results
- **Model Accuracy:** 77% using Random Forest Classifier
- **Churn Rate:** 26.5% of customers churned
- **Top Churn Drivers:** Monthly Charges, Total Charges, Tenure

## 💡 Key Insights
| Finding | Insight |
|---------|---------|
| Month-to-month contracts | Highest churn (1,655 customers) |
| Two-year contracts | Lowest churn (only 48 customers) |
| Electronic check users | Churn the most (17,954) |
| New customers (avg 18 months) | Leave faster than loyal customers (avg 37 months) |
| Higher monthly charges ($74) | More likely to churn vs $61 for loyal customers |

## 🛠️ Tools & Technologies
- **Python:** Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib
- **Power BI:** Interactive dashboard with 5 visualizations
- **Tableau:** Interactive dashboard with 5 visualizations
- **Machine Learning:** Random Forest Classifier

## 📁 Files
| File | Description |
|------|-------------|
| `Churn_Analysis_Complete.py` | Full Python code (cleaning, EDA, model) |
| `Telco_Churn_Clean.csv` | Cleaned dataset |
| `Customer_Churn_Dashboard.pbix` | Power BI dashboard |
| `Customer_Churn_Dashboard.twbx` | Tableau dashboard |
| `01_churn_pie.png` | Churn distribution chart |
| `02_churn_by_contract.png` | Churn by contract type |
| `03_tenure_by_churn.png` | Tenure analysis |
| `04_charges_by_churn.png` | Monthly charges analysis |
| `05_churn_by_payment.png` | Payment method analysis |

## 📊 Dashboard Preview

### Power BI
<img width="1169" height="656" alt="image" src="https://github.com/user-attachments/assets/4f9cd543-4b77-480e-8b89-0fda8ab2956b" />



### Tableau
<img width="638" height="851" alt="image" src="https://github.com/user-attachments/assets/62ea0859-0916-4e31-a327-c2541c914438" />



## 🔍 Analysis Process
1. **Data Cleaning:** Fixed missing values, converted data types
2. **EDA:** Created 5 visualizations to understand patterns
3. **Feature Engineering:** Created binary target variable
4. **Model Building:** Random Forest with 77% accuracy
5. **Dashboards:** Built interactive dashboards in Power BI & Tableau

## 📈 Feature Importance
| Feature | Importance |
|---------|------------|
| Monthly Charges | 38.6% |
| Total Charges | 35.3% |
| Tenure | 23.3% |
| Senior Citizen | 2.8% |

## 👩‍💻 Author
**Maha Pentakota**
- LinkedIn: [linkedin.com/in/mahapentakota](https://linkedin.com/in/mahapentakota)
- GitHub: [github.com/Maha112024](https://github.com/Maha112024)

## 📜 License
This project is open source and available for learning purposes.
