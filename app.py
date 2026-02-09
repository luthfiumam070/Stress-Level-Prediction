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

label_map_kualitas = {
    1: "Sangat Buruk",
    2: "Buruk",
    3: "Sedang",
    4: "Baik",
    5: "Sangat Baik"
}

label_map_frekuensi = {
    1: "Tidak Pernah",
    2: "Sangat Jarang",
    3: "Sedang",
    4: "Sering",
    5: "Sangat Sering"
}

label_map_Beban = {
    1: "Tidak Berat",
    2: "Sedikit Berat",
    3: "Sedang",
    4: "Berat",
    5: "Sangat Berat"
}

sleep = st.slider(
    "Kualitas Tidur",
    min_value=1,
    max_value=5,
    value=3,
    format_func=lambda x: label_map_kualitas[x]
)

headache = st.slider(
    "Frekuensi Sakit Kepala",
    min_value=1,
    max_value=5,
    value=3,
    format_func=lambda x: label_map_frekuensi[x]
)
academic = st.slider(
    "Performa Akademik",
    min_value=1,
    max_value=5,
    value=3,
    format_func=lambda x: label_map_kualitas[x]
)
study_load = st.slider(
    "Beban Studi",
    min_value=1,
    max_value=5,
    value=3,
    format_func=lambda x: label_map_Beban[x]
)
extracurricular = st.slider(
    "Beban Kegiatan Ekstrakurikuler", 
    min_value=1,
    max_value=5,
    value=3,
    format_func=lambda x: label_map_Beban[x]
)

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
