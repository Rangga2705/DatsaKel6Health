import streamlit as st
import pandas as pd
import joblib

# Load trained model (sesuaikan nama file model kamu)
model = joblib.load('linear_regression_model.pkl')

st.title('Prediksi Data Kesehatan')
st.write('Aplikasi ini memprediksi nilai kesehatan berdasarkan parameter pengguna.')

# Sidebar input
st.sidebar.header("Input Parameter Kesehatan")

def user_input_features():
    age = st.sidebar.slider('Usia (tahun)', 10, 80, 30)
    height = st.sidebar.slider('Tinggi Badan (cm)', 120, 200, 165)
    weight = st.sidebar.slider('Berat Badan (kg)', 30, 120, 60)
    bmi = st.sidebar.slider('BMI', 10.0, 40.0, 22.5)
    heart_rate = st.sidebar.slider('Detak Jantung (bpm)', 50, 150, 85)
    sleep_hours = st.sidebar.slider('Jam Tidur (per Hari)', 1, 12, 7)

    data = {
        'Age': age,
        'Height': height,
        'Weight': weight,
        'BMI': bmi,
        'Heart_Rate': heart_rate,
        'Sleep_Hours': sleep_hours
    }

    return pd.DataFrame(data, index=[0])

df_input = user_input_features()

st.subheader("Parameter Input Pengguna:")
st.write(df_input)

# Prediction Button
if st.sidebar.button("Prediksi"):
    try:
        prediction = model.predict(df_input)
        st.subheader("Hasil Prediksi:")
        st.write(f"Nilai Prediksi: {prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"Error saat prediksi: {e}")
        st.exception(e)
