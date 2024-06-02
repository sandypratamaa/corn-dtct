import streamlit as st
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from PIL import Image
from werkzeug.utils import secure_filename
import base64

# Set nilai default untuk hasil prediksi dan gambar yang diprediksi
hasil_prediksi = '(none)'
gambar_prediksi = '(none)'

# Define classes
corndiseases_classes = ["Corn Common Rust", "Corn Gray Leaf Spot", "Corn Healthy", "Corn Northern Leaf Blight", "NON DETECT"]

# Load model with error handling
model_path = "corn_model.h5"

model = None
if os.path.exists(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"Model file not found at {model_path}")

# Set Streamlit configuration
st.set_page_config(page_title="Corn Disease Detection", page_icon=":corn:", layout="wide")

# Sidebar
st.sidebar.title("Corn Disease Detection")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg"])

# Main content
st.title("Welcome to Corn Disease Detection :corn:")
st.markdown("*Aplikasi ini dapat membantu dalam mengklasifikasi kondisi tanaman jagung anda*")

# Define image size
IMG_SIZE = (299, 299)
st.image(image="ss.png", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

# Explanation
st.markdown("Aplikasi ini berguna untuk mendeteksi penyakit pada tanaman jagung menggunakan teknologi kecerdasan buatan (AI) dan algoritma deep learning untuk mendiagnosis penyakit pada tanaman jagung melalui gambar yang diunggah ke aplikasi ini. Dataset yang digunakan dalam sistem ini terdiri dari ribuan gambar tanaman jagung yang terinfeksi penyakit dan sehat. Saat pengguna mengunggah gambar tanaman jagung, sistem akan menganalisis gambar tersebut dan memberikan diagnosis. Algoritma deep learning digunakan karena dapat mempelajari fitur-fitur kompleks yang terkait dengan penyakit pada tanaman jagung dan menghasilkan diagnosis yang lebih akurat.")
st.markdown(":corn: Terdapat 4 jenis kategori yang aplikasi dapat deteksi yaitu Corn Common Rust, Corn Northern Leaf Blight, Corn Gray Leaf Spot, dan Corn Healthy yang akan di proses dibawah ini : ")

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    test_image = Image.open(uploaded_file).resize(IMG_SIZE)
    img_array = np.expand_dims(test_image, 0) / 255.0  # Normalize the image

    # Predict if model is loaded
    if model:
        predictions = model.predict(img_array)
        hasil_prediksi = corndiseases_classes[np.argmax(predictions[0])]
        st.success(f"Prediction: {hasil_prediksi}")
            else:
                 st.error("Model is not loaded. Unable to make predictions.")

st.subheader("Penjelasan mengenai jenis-jenis penyakit pada tanaman jagung")

st.markdown("1. **Corn Common Rust** atau karat jagung adalah penyakit yang disebabkan oleh jamur *Puccinia sorghi*. Penyakit ini umum terjadi pada tanaman jagung di berbagai daerah dengan iklim yang hangat dan lembap. Gejalanya meliputi adanya bercak-bercak berwarna kuning atau oranye pada daun tanaman jagung. Infeksi karat jagung biasanya tidak menyebabkan kerusakan yang serius pada hasil panen, tetapi dapat mengurangi pertumbuhan dan produktivitas tanaman jika serangan parah terjadi.")
st.markdown("2. **Corn Gray Leaf Spot** atau bercak daun abu-abu pada jagung disebabkan oleh jamur *Cercospora zeae-maydis*. Penyakit ini biasanya terjadi pada pertengahan hingga akhir musim tanam dan lebih umum terjadi di daerah yang lembap. Gejala utamanya adalah adanya bercak-bercak berwarna abu-abu atau coklat kehitaman pada daun jagung. Serangan berat dapat menyebabkan penurunan produksi dan kualitas jagung.")
st.markdown("3. **Corn Northern Leaf Blight** atau bercak daun utara pada jagung disebabkan oleh jamur *Exserohilum turcicum*. Penyakit ini biasanya terjadi pada musim panas yang lembap dan hangat. Gejalanya meliputi adanya bercak-bercak berwarna coklat atau hijau keabu-abuan pada daun tanaman jagung. Serangan yang parah dapat menyebabkan kerusakan pada daun, mengurangi efisiensi fotosintesis, dan berpotensi mengurangi hasil panen.")
st.markdown("4. **Corn Healthy** adalah kondisi bahwa tanaman jagung anda dalam kondisi sehat.")
