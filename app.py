import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load('knn_stress_model.pkl')
scaler = joblib.load("scaler.pkl")

# =========================
# UI
# =========================
st.set_page_config(
    page_title="Prediksi Stres Mahasiswa",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Prediksi Tingkat Stres Mahasiswa")
st.caption("Model: K-Nearest Neighbor")
st.write(
    "Aplikasi ini memprediksi tingkat stres mahasiswa "
    "berdasarkan kuesioner menggunakan model **KNN**."
)

st.divider()

# =========================
# INPUT
# =========================
sleep = st.slider("Kualitas Tidur", 1, 5, 3)
headache = st.slider("Frekuensi Sakit Kepala", 1, 5, 3)
academic = st.slider("Performa Akademik", 1, 5, 3)
study_load = st.slider("Beban Studi", 1, 5, 3)
extracurricular = st.slider("Beban Kegiatan Ekstrakurikuler", 1, 5, 3)

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

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    st.divider()
    st.subheader("Hasil Prediksi")

    if prediction == 1:
        st.success(f"😊 Tingkat Stres Sangat Rendah ({prediction})")
    elif prediction == 2:
        st.warning(f"😊 Tingkat Stres Rendah ({prediction})")
    elif prediction == 3:
        st.warning(f"😐 Tingkat Stres Sedang ({prediction})")
    elif prediction == 4:
        st.warning(f"😟 Tingkat Stres Tinggi ({prediction})")
    else:
        st.error(f"😟 Tingkat Stres Sangat Tinggi ({prediction})")
