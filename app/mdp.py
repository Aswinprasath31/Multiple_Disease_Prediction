import streamlit as st
import numpy as np
import pickle

# Load models and scalers
kidney_model = pickle.load(open('kidney_model.pkl', 'rb'))
kidney_scaler = pickle.load(open('kidney_scaler.pkl', 'rb'))

liver_model = pickle.load(open('liver_model.pkl', 'rb'))
liver_scaler = pickle.load(open('liver_scaler.pkl', 'rb'))

parkinsons_model = pickle.load(open('parkinsons_model.pkl', 'rb'))
parkinsons_scaler = pickle.load(open('parkinsons_scaler.pkl', 'rb'))

# Page title
st.title("🧬 Multiple Disease Prediction App")
st.sidebar.title("Choose Disease")

# Disease selection
disease = st.sidebar.selectbox("Select a Disease", ["Kidney Disease", "Liver Disease", "Parkinson's Disease"])

# ========== KIDNEY FORM ==========
if disease == "Kidney Disease":
    st.header("Kidney Disease Prediction")

    age = st.number_input("Age")
    bp = st.number_input("Blood Pressure")
    sg = st.number_input("Specific Gravity", format="%.2f")
    al = st.number_input("Albumin")
    su = st.number_input("Sugar")
    bgr = st.number_input("Blood Glucose Random")
    bu = st.number_input("Blood Urea")
    sc = st.number_input("Serum Creatinine")
    sod = st.number_input("Sodium")
    pot = st.number_input("Potassium")
    hemo = st.number_input("Hemoglobin")
    pcv = st.number_input("Packed Cell Volume")
    wc = st.number_input("White Blood Cell Count")
    rc = st.number_input("Red Blood Cell Count")

    input_data = [age, bp, sg, al, su, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc]
    if st.button("Predict"):
        input_scaled = kidney_scaler.transform([input_data])
        result = kidney_model.predict(input_scaled)[0]
        prob = kidney_model.predict_proba(input_scaled)[0][1]
        st.success(f"Disease Prediction: {'CKD' if result==1 else 'Normal'}")
        st.info(f"Probability of disease: {prob:.2f}")

# ========== LIVER FORM ==========
elif disease == "Liver Disease":
    st.header("Liver Disease Prediction")

    age = st.number_input("Age")
    gender = st.selectbox("Gender", ["Male", "Female"])
    tb = st.number_input("Total Bilirubin")
    db = st.number_input("Direct Bilirubin")
    alkphos = st.number_input("Alkaline Phosphotase")
    sgpt = st.number_input("Alamine Aminotransferase")
    sgot = st.number_input("Aspartate Aminotransferase")
    tp = st.number_input("Total Proteins")
    alb = st.number_input("Albumin")
    ag_ratio = st.number_input("A/G Ratio")

    gender_encoded = 1 if gender == "Male" else 0

    input_data = [age, gender_encoded, tb, db, alkphos, sgpt, sgot, tp, alb, ag_ratio]
    if st.button("Predict"):
        input_scaled = liver_scaler.transform([input_data])
        result = liver_model.predict(input_scaled)[0]
        prob = liver_model.predict_proba(input_scaled)[0][1]
        st.success(f"Disease Prediction: {'Liver Disease' if result==1 else 'Normal'}")
        st.info(f"Probability of disease: {prob:.2f}")

# ========== PARKINSONS FORM ==========
elif disease == "Parkinson's Disease":
    st.header("Parkinson's Disease Prediction")

    fo = st.number_input("MDVP:Fo(Hz)")
    fhi = st.number_input("MDVP:Fhi(Hz)")
    flo = st.number_input("MDVP:Flo(Hz)")
    jitter_percent = st.number_input("MDVP:Jitter(%)")
    jitter_abs = st.number_input("MDVP:Jitter(Abs)")
    rap = st.number_input("MDVP:RAP")
    ppq = st.number_input("MDVP:PPQ")
    ddp = st.number_input("Jitter:DDP")
    shimmer = st.number_input("MDVP:Shimmer")
    shimmer_db = st.number_input("MDVP:Shimmer(dB)")
    apq3 = st.number_input("Shimmer:APQ3")
    apq5 = st.number_input("Shimmer:APQ5")
    apq = st.number_input("MDVP:APQ")
    dda = st.number_input("Shimmer:DDA")
    nhr = st.number_input("NHR")
    hnr = st.number_input("HNR")
    rpde = st.number_input("RPDE")
    dfa = st.number_input("DFA")
    spread1 = st.number_input("spread1")
    spread2 = st.number_input("spread2")
    d2 = st.number_input("D2")
    ppe = st.number_input("PPE")

    input_data = [fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp, shimmer, shimmer_db,
                  apq3, apq5, apq, dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]

    if st.button("Predict"):
        input_scaled = parkinsons_scaler.transform([input_data])
        result = parkinsons_model.predict(input_scaled)[0]
        prob = parkinsons_model.predict_proba(input_scaled)[0][1]
        st.success(f"Disease Prediction: {'Parkinson’s' if result==1 else 'Normal'}")
        st.info(f"Probability of disease: {prob:.2f}")
