import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

# App titel
st.title("Hartziekte voorspeller")
st.write("Vul patiëntwaarden in. De app geeft een kans en een advies (doorverwijzen ja/nee).")

# Laad model
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "heart_model.joblib"
clf = joblib.load(MODEL_PATH)

# Instelbare drempel (threshold)
threshold = st.slider("Drempel voor doorverwijzen (kans >= drempel)", 0.0, 1.0, 0.50, 0.01)

st.subheader("Patiëntgegevens")

# Invoer velden (numeriek)
age = st.number_input("Age (jaren)", min_value=0, max_value=120, value=55)
restingbp = st.number_input("RestingBP (mm Hg)", min_value=0, max_value=300, value=145)
chol = st.number_input("Cholesterol (mg/dl)", min_value=0, max_value=1000, value=220)
maxhr = st.number_input("MaxHR", min_value=60, max_value=202, value=140)
oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=1.2, step=0.1)

fastingbs = st.selectbox("FastingBS > 120 mg/dl?", options=[0, 1], index=0)
restingecg = st.selectbox("RestingECG", options=[0, 1, 2], index=0)

# Invoer velden (categorisch)
sex = st.selectbox("Sex", options=["M", "F"], index=0)
cpt = st.selectbox("ChestPainType", options=["TA", "ATA", "NAP", "ASY"], index=1)
exang = st.selectbox("ExerciseAngina", options=["N", "Y"], index=0)
stslope = st.selectbox("ST_Slope", options=["Up", "Flat", "Down"], index=1)

# Knop
if st.button("Voorspel"):
    new_patient = pd.DataFrame([{
        "Age": age,
        "RestingBP": restingbp,
        "Cholesterol": chol,
        "MaxHR": maxhr,
        "Oldpeak": oldpeak,
        "FastingBS": fastingbs,
        "RestingECG": restingecg,
        "Sex": sex,
        "ChestPainType": cpt,
        "ExerciseAngina": exang,
        "ST_Slope": stslope
    }])

    prob = clf.predict_proba(new_patient)[0, 1]
    pred = int(prob >= threshold)

    st.markdown(f"### Kans op hartziekte: **{prob:.3f}**")
    if pred == 1:
        st.error("Advies: **Doorverwijzen** (hoge kans op hartziekte).")
    else:
        st.success("Advies: **Geen doorverwijzing** (lage kans op hartziekte).")