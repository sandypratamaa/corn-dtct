import streamlit as st
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from PIL import Image
from werkzeug.utils import secure_filename
import base64
import logging

# Set nilai default untuk hasil prediksi dan gambar yang diprediksi
hasil_prediksi = '(none)'
gambar_prediksi = '(none)'

# Load model
#model = tf.keras.models.load_model("corn_model_done.h5")

logging.basicConfig(level=logging.DEBUG)

model_path = "corn_model_done.h5"
if not os.path.exists(model_path):
    st.error(f"Model file {model_path} does not exist.")
    logging.error(f"Model file {model_path} does not exist.")
else:
    try:
        model = tf.keras.models.load_model(model_path)
        st.success("Model loaded successfully.")
    except OSError as e:
        st.error(f"Failed to load model: {str(e)}")
        logging.error("Failed to load model", exc_info=True)

# Define classes
corndiseases_classes = ["Corn Common Rust", "Corn Gray Leaf Spot", "Corn Healthy", "Corn Northern Leaf Blight", "Non detect"]

# Set Streamlit configuration
st.set_page_config(page_title="Corn Disease Detection", page_icon=":corn:", layout="wide")

# Sidebar
st.sidebar.title("Corn Disease Detection")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg"])

# Main content
st.title("Welcome to Corn Disease Detection :corn:")
st.markdown("*Aplikasi ini dapat membantu dalam mengklasifikasi kondisi tanaman jagung anda*")

#add image
# Define image size
IMG_SIZE = (299, 299)
st.image(image="ss.png", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

#penjelasan
st.markdown("Aplikasi ini berguna untuk mendeteksi penyakit pada tanaman jagung menggunakan teknologi kecerdasan buatan (AI) dan algoritma deep learning untuk mendiagnosis penyakit pada tanaman jagung melalui gambar yang diunggah ke aplikasi ini. Dataset yang digunakan dalam sistem ini terdiri dari ribuan gambar tanaman jagung yang terinfeksi penyakit dan sehat. Saat pengguna mengunggah gambar tanaman jagung, sistem akan menganalisis gambar tersebut dan memberikan diagnosis. Algoritma deep learning digunakan karena dapat mempelajari fitur-fitur kompleks yang terkait dengan penyakit pada tanaman jagung dan menghasilkan diagnosis yang lebih akurat.")

st.markdown(":corn: Terdapat 4 jenis kategori yang aplikasi dapat deteksi yaitu Corn Common Rust, Corn Northern Leaf Blight, Corn Gray Leaf Spot, dan Corn Healty yang akan di proses dibawah ini : ")

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Predict
  
    test_image = Image.open(uploaded_file).resize(IMG_SIZE)
    img_array = np.expand_dims(test_image, 0)

    predictions = model.predict(img_array)
    hasil_prediksi = corndiseases_classes[np.argmax(predictions[0])]

    # Display result
    st.success(f"Prediction: {hasil_prediksi}")

st.subheader(" Penjelasan mengenai jenis-jenis penyakit pada tanaman jagung ")

st.markdown("1.Corn Common Rust atau karat jagung adalah penyakit yang disebabkan oleh jamur Puccinia sorghi. Penyakit ini umum terjadi pada tanaman jagung di berbagai daerah dengan iklim yang hangat dan lembap. Gejalanya meliputi adanya bercak-bercak berwarna kuning atau oranye pada daun tanaman jagung. Infeksi karat jagung biasanya tidak menyebabkan kerusakan yang serius pada hasil panen, tetapi dapat mengurangi pertumbuhan dan produktivitas tanaman jika serangan parah terjadi.")
st.markdown("2.Corn Gray Leaf Spot atau bercak daun abu-abu pada jagung disebabkan oleh jamur Cercospora zeae-maydis. Penyakit ini biasanya terjadi pada pertengahan hingga akhir musim tanam dan lebih umum terjadi di daerah yang lembap. Gejala utamanya adalah adanya bercak-bercak berwarna abu-abu atau coklat kehitaman pada daun jagung. Serangan berat dapat menyebabkan penurunan produksi dan kualitas jagung.")
st.markdown("3.Corn Northern Leaf Blight atau bercak daun utara pada jagung disebabkan oleh jamur Exserohilum turcicum. Penyakit ini biasanya terjadi pada musim panas yang lembap dan hangat. Gejalanya meliputi adanya bercak-bercak berwarna coklat atau hijau keabu-abuan pada daun tanaman jagung. Serangan yang parah dapat menyebabkan kerusakan pada daun, mengurangi efisiensi fotosintesis, dan berpotensi mengurangi hasil panen.")
st.markdown("4.Corn Healty adalah kondisi bahwa tanaman jagung anda dalam kondisi sehat.")
