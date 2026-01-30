# 🩺 Diabetes Prediction System (Capstone Project)

This project is a **Diabetes Risk Prediction Web Application** built using **Classical Machine Learning** techniques and deployed with an interactive **Streamlit** interface.

It was developed as a **Capstone Project** during my Data Science certification program.

---

## 🚀 Project Overview

Early detection of diabetes is important for preventive healthcare.  
This application predicts the probability of Type 2 diabetes risk based on key diagnostic medical inputs such as glucose level, BMI, insulin, age, etc.

The system uses a **Logistic Regression** model trained on the **Pima Indians Diabetes Dataset**.

---

## 🔄 Workflow Diagram

![Workflow](images/workflow.png)

*(UI icons and visual assets used in the Streamlit app are also included in the `images/` folder.)*

---

## ✨ Key Features

- 📊 Predicts diabetes risk with probability score  
- 🧹 Data preprocessing includes:
  - Medical range filtering for plausible values  
  - Median imputation for invalid zero values  
  - Feature scaling using StandardScaler  
- 🤖 Machine Learning Models:
  - Logistic Regression (Preferred for screening)
  - Linear Discriminant Analysis (Comparison)
- 🖥 Interactive Streamlit Dashboard:
  - User-friendly sliders for medical input  
  - Risk-based prediction output  
  - Lifestyle and diet guideline suggestions  
- ⚠️ Includes medical disclaimer for responsible use  

---

## 📈 Model Performance (Summary)

- Accuracy: ~79%  
- ROC–AUC Score: ~0.85  
- Logistic Regression Recall (Diabetes class): ~76.9%  
- Logistic Regression was selected over LDA due to better sensitivity for screening tasks.

---

## 📄 Project Report

A detailed methodology and evaluation report is available here:

📌 **docs/project_report.pdf**

---

## 🛠 Tech Stack

- **Python**
- **Pandas, NumPy**
- **Scikit-learn**
  - Logistic Regression
  - Linear Discriminant Analysis
  - StandardScaler
- **Streamlit** (Web UI)

---

## 📂 Project Structure

```bash
diabetes-prediction-streamlit/
│
├── app.py
├── Diabetes_Prediction.ipynb
├── diabetes.csv
├── requirements.txt
│
├── images/
│   ├── banner.png
│   ├── warning.png
│   ├── success.png
│   ├── low_risk.png
│   ├── moderate_risk.png
│   ├── high_risk.png
│   └── workflow_diagram.png
│
└── docs/
    └── project_report.pdf
