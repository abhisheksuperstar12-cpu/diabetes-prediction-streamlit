import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Diabetes Prediction App",
    layout="centered"
)

# ---------------- SAFE IMAGE FUNCTION ----------------
from streamlit.runtime.media_file_storage import MediaFileStorageError

def safe_image(path, width=None):
    """
    Display an image if it exists.
    If the file is missing or cannot be read, do nothing.
    """
    try:
        st.image(path, width=width)
    except (FileNotFoundError, MediaFileStorageError):
        pass

# ---------------- HEADER BANNER ----------------
import base64

def show_banner(image_path):
    """
    Display banner image if it exists, else skip silently.
    """
    try:
        with open(image_path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode()
        st.markdown(
            f"""
            <style>
            .banner {{
                width: 100%;
                max-width: 1200px;
                max-height: 180px;
                object-fit: contain;
                display: block;
                margin-left: auto;
                margin-right: auto;
            }}
            </style>

            <img class="banner" src="data:image/png;base64,{encoded}">
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        pass

show_banner("images/banner.png")

st.title("🩺 Diabetes Prediction System 💙")
st.markdown(
    """
    ### 🧪 Predict • 🧠 Understand • ❤️ Prevent
    <br><br>
    """,
    unsafe_allow_html=True
)

# ---------------- LOAD DATA ----------------
df = pd.read_csv("diabetes.csv")

# -------- REMOVE MEDICALLY INVALID OUTLIERS --------
medical_ranges = {
    "Pregnancies": (0, 20),
    "Glucose": (70, 200),
    "BloodPressure": (40, 140),
    "SkinThickness": (5, 100),
    "Insulin": (15, 900),
    "BMI": (10, 60),
    "DiabetesPedigreeFunction": (0.05, 3.0),
    "Age": (18, 100)
}

for col, (low, high) in medical_ranges.items():
    df = df[(df[col] >= low) & (df[col] <= high)]

# ---------------- MEDIAN IMPUTATION ----------------
cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_with_zero:
    df[col] = df[col].replace(0, df[col].median())

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# ---------------- SCALING ----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------- MODEL ----------------
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)
model.fit(X_scaled, y)

# ---------------- SIDEBAR INPUT ----------------
st.sidebar.caption(
    "⚠️ This model estimates Type 2 diabetes risk for adults (18+). "
    "It is not intended for children or Type 1 diabetes diagnosis."
)

def user_input():
    pregnancies = st.sidebar.number_input("🤰 Pregnancies", 0, 20, 1)
    glucose = st.sidebar.number_input("🩸 Glucose", 50, 200, 120)
    bp = st.sidebar.number_input("💓 Blood Pressure", 40, 140, 70)
    skin = st.sidebar.number_input("📏 Skin Thickness", 0, 100, 20)
    insulin = st.sidebar.number_input("💉 Insulin", 0, 900, 80)
    bmi = st.sidebar.number_input("⚖️ BMI", 10.0, 60.0, 25.0)
    dpf = st.sidebar.number_input("🧬 Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age = st.sidebar.number_input("🎂 Age", 18, 100, 30)

    return pd.DataFrame(
        [[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]],
        columns=X.columns
    )

input_df = user_input()

# Replace any zero entered by user with median of training data to remove any zero values inconsistency
for col in cols_with_zero:
    if col in input_df.columns:    
        if input_df[col].iloc[0] == 0:
            input_df.at[0, col] = df[col].median()
input_scaled = scaler.transform(input_df)

# ---------------- INPUT EXPLANATION ----------------
with st.expander("ℹ️ What do these inputs mean?"):
    st.markdown(""" 
    - **Glucose**: Blood sugar level after fasting  
    - **BMI**: Body Mass Index  
    - **DPF**: Genetic risk indicator  
    - **Insulin**: Blood insulin level  
    - **Blood Pressure**: Systolic/Diastolic pressure  
    - **Skin Thickness**: Triceps skin fold thickness  
    - **Pregnancies**: Number of times pregnant  
    - **Age**: Patient's age in years  
    """)

# ---------------- PREDICTION ----------------
prediction = model.predict(input_scaled)[0]
probability = model.predict_proba(input_scaled)[0][1]

st.subheader("🔍 Prediction Result 🧾")

if prediction == 1:
    safe_image("images/warning.png", width=60)
    st.error(f"🚨 High Diabetes Risk Detected!\n\n📊 Probability: {probability * 100:.1f}%")
    st.markdown("👉 **Please consult a doctor for further evaluation.**")
else:
    safe_image("images/success.png", width=60)
    st.success(f"🎉 Low Diabetes Risk\n\n📊 Probability: {probability * 100:.1f}%")
    st.markdown("👍 **Maintain a healthy lifestyle!**")

st.caption(
    "Note: Prediction confidence is based on a logistic regression model; trained on 'Pima Indians Diabetes Database'.")

# ---------------- GUIDELINES ----------------
st.markdown("---")
st.subheader("🧠 Lifestyle & Diet Guidelines 🍎🥗")

st.caption(
    "⚠️ **Disclaimer:** These suggestions are for general awareness only 📘. "
    "They do NOT replace professional medical advice 🩺."
)

risk_level = st.selectbox(
    "📊 Select Diabetes Risk Probability Range",
    [
        "Below 30% (Low Risk)",
        "30% – 60% (Moderate Risk)",
        "Above 60% (High Risk)"
    ]
)

if risk_level == "Below 30% (Low Risk)":
    safe_image("images/low_risk.png", width=100)
    st.success("🟢 Low Risk – Indian Diet & Lifestyle Tips 🇮🇳")
    st.markdown("""
    🥗 Eat whole grains (roti, millets, brown rice)  
    🚶 Walk or do yoga for 30 minutes daily  
    🍎 Include fruits & vegetables  
    🍬 Limit sweets and sugary drinks  
    🩺 Regular health checkups  
    """)

elif risk_level == "30% – 60% (Moderate Risk)":
    safe_image("images/moderate_risk.png", width=100)
    st.warning("🟡 Moderate Risk – Indian Diet & Lifestyle Tips 🇮🇳")
    st.markdown("""
    🍚 Replace white rice with brown rice/millets  
    🚫 Avoid fried snacks (samosa, pakora)  
    🥬 Increase fiber-rich foods  
    ☕ Reduce sugar in tea/coffee  
    📅 Monitor glucose levels  
    """)

else:
    safe_image("images/high_risk.png", width=100)
    st.error("🔴 High Risk – Indian Diet & Lifestyle Tips 🇮🇳")
    st.markdown("""
    🚨 Consult a doctor immediately  
    🍰 Avoid sweets, desserts, bakery items  
    🥦 Prefer low-GI foods  
    📉 Monitor blood sugar regularly  
    📋 Follow doctor-prescribed diet  
    """)

st.markdown("---")
st.caption(
    "This application is for educational purposes only and is not a diagnostic tool.")
st.caption("App Version 1.0 • ML Model: Logistic Regression")
