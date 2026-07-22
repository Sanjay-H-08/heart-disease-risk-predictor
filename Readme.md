# 🫀 Heart Disease Prediction using Custom Logistic Regression
**🌟 Try the live app here:** [Click to open Heart Disease Predictor](https://heart-disease-risk-predictor08.streamlit.app/)
## 🎯 Overview
This project predicts the likelihood of heart disease based on clinical patient data. Rather than using pre-built black-box estimators, it features a **custom Logistic Regression model built from scratch** using NumPy, evaluated with custom evaluation tools, and deployed via an interactive **Streamlit** web app.

---

## 📊 Key EDA Insights
Through Exploratory Data Analysis on the `heart.csv` dataset, we identified several critical clinical markers:
* 🟢 **Zero Missing Data:** All patient records are complete and validated.
* 📈 **Strong Positive Predictors:** Chest pain type (`cp`) and maximum heart rate achieved (`thalach`) correlate most strongly with a positive heart disease diagnosis.
* 📉 **Inverse Risk Markers:** Exercise-induced angina (`exang`) and ST depression (`oldpeak`) show significant negative correlation with the target.

---

## 🛠️ Tech Stack & Methods
* **Data Handling & EDA:** `pandas`, `seaborn`, `matplotlib` 📈
* **Model Implementation:** Custom `LogisticRegression` class (Gradient Descent & Sigmoid activation) 🧠
* **Data Processing:** Custom train/test split and Feature Scaling (`StandardScaler`) ⚖️
* **Deployment:** `Streamlit` 🚀

---

## 🚀 How to Run

1. **Clone the Repository & Install Dependencies:**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn streamlit
