import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn import tree

# Konfigurasi halaman dengan tema aesthetic
st.set_page_config(
    page_title="Prediksi Risiko Gagal Jantung",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load model
loaded_model = joblib.load('cart_model_heartz.pkl')

# Gaya CSS untuk tampilan aesthetic
st.markdown("""
    <style>
        /* Warna latar belakang */
        body {
            background-color: #f7f1e3;
            color: #2f3542;
            font-family: "Arial", sans-serif;
        }
        h1, h2, h3 {
            color: #57606f;
            font-weight: 700;
        }
        .stButton button {
            background-color: #ffa502;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 16px;
            font-weight: bold;
        }
        .stButton button:hover {
            background-color: #ff7f50;
            transition: 0.3s;
        }
        .sidebar .stRadio {
            background-color: #dfe4ea;
            border-radius: 10px;
            padding: 10px;
        }
        .stNumberInput, .stSelectbox {
            background-color: #ffffff;
            border: 1px solid #dcdde1;
            border-radius: 8px;
        }
        .stPlotlyChart {
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

def main():
    # Sidebar navigasi
    st.sidebar.header("Menu")
    page = st.sidebar.radio("Pilih Halaman", ["Input Data", "Visualisasi Model"])

    if page == "Input Data":
        st.title("🩺 Prediksi Risiko Gagal Jantung")
        st.markdown(
            """
            Selamat datang! 🎉 Aplikasi ini menggunakan **model CART** untuk memprediksi apakah pasien berisiko mengalami gagal jantung.
            Harap isi data dengan lengkap di bawah ini.  
            """,
            unsafe_allow_html=True,
        )

        # Input data pasien
        st.subheader("📝 Masukkan Data Pasien")

        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Umur (Tahun)", min_value=0, max_value=120, value=50)
            anemia = st.selectbox("Anemia", ["0 = Tidak", "1 = Ya"])
            cpk = st.number_input("Creatinine Phosphokinase (CPK)", min_value=0, value=100)
            diabetes = st.selectbox("Diabetes", ["0 = Tidak", "1 = Ya"])
            ef = st.number_input("Ejection Fraction (EF)", min_value=0, max_value=100, value=50)
            hbp = st.selectbox("Tekanan Darah Tinggi", ["0 = Tidak", "1 = Ya"])

        with col2:
            platelets = st.number_input("Trombosit", min_value=0.0, value=200000.0)
            serum_creatinine = st.number_input("Serum Creatinine", min_value=0.0, value=1.0)
            serum_sodium = st.number_input("Serum Sodium", min_value=0, value=140)
            sex = st.selectbox("Jenis Kelamin", ["0 = Wanita", "1 = Pria"])
            smoking = st.selectbox("Perokok", ["0 = Tidak", "1 = Ya"])
            time = st.number_input("Waktu Diagnosa (Hari)", min_value=0, value=30)

        # Konversi input
        inputs = [
            age,
            1 if anemia == "Ya" else 0,
            cpk,
            1 if diabetes == "Ya" else 0,
            ef,
            1 if hbp == "Ya" else 0,
            platelets,
            serum_creatinine,
            serum_sodium,
            1 if sex == "Pria" else 0,
            1 if smoking == "Ya" else 0,
            time,
        ]

        # Tombol prediksi
        if st.button("Prediksi Risiko"):
            prediction = loaded_model.predict([inputs])[0]
            outcome = "⚠️ Pasien Diprediksi Meninggal" if prediction == 1 else "✅ Pasien Diprediksi Selamat"
            color = "red" if prediction == 1 else "green"

            # Tampilkan hasil prediksi
            st.markdown(
                f"<h2 style='color: {color}; text-align: center;'>{outcome}</h2>",
                unsafe_allow_html=True,
            )

    elif page == "Visualisasi Model":
        st.title("🔍 Visualisasi Pohon Keputusan")
        st.write("Berikut adalah visualisasi dari model CART yang digunakan:")

        feature_names = [
            "Age",
            "Anaemia",
            "Creatinine Phosphokinase",
            "Diabetes",
            "Ejection Fraction",
            "High Blood Pressure",
            "Platelets",
            "Serum Creatinine",
            "Serum Sodium",
            "Sex",
            "Smoking",
            "Time",
        ]
        target_name= ["alive","death"]

        fig = plt.figure(figsize=(20,15))
        _ = tree.plot_tree(loaded_model,
                feature_names = feature_names,
                class_names = target_name,
                filled = True,
                rounded=True,  # Menambahkan sudut melengkung di node
                fontsize=8)  # Menyesuaikan ukuran font yang lebih kecil)
        
        st.pyplot(fig)
if __name__ == "__main__":
    main()
