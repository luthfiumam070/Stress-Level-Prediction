import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load('knn_stress_model.pkl')

# =========================
# UI
# =========================
st.set_page_config(
    page_title="Prediksi Stres Mahasiswa",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Prediksi Tingkat Stres Mahasiswa")
st.write(
    "Aplikasi ini memprediksi tingkat stres mahasiswa "
    "berdasarkan kuesioner menggunakan model **KNN**."
)

st.divider()

# =========================
# INPUT
# =========================
sleep = st.slider("Sleep Quality", 1, 5, 3)
headache = st.slider("Headache Frequency per Week", 1, 5, 3)
academic = st.slider("Academic Performance", 1, 5, 3)
study_load = st.slider("Study Load", 1, 5, 3)
extracurricular = st.slider("Extracurricular Activities", 1, 5, 3)

# =========================
# PREDICT
# =========================
if st.button("🔍 Prediksi Tingkat Stres"):
    input_data = np.array([[
        sleep,
        headache,
        academic,
        study_load,
        extracurricular
    ]])

    prediction = model.predict(input_data)[0]

    st.divider()
    st.subheader("Hasil Prediksi")

    if prediction <= 2:
        st.success(f"😊 Tingkat Stres Rendah ({prediction})")
    elif prediction == 3:
        st.warning(f"😐 Tingkat Stres Sedang ({prediction})")
    else:
        st.error(f"😟 Tingkat Stres Tinggi ({prediction})")