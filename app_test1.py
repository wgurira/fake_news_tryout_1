pip install tensorflow
import streamlit as st
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import requests
from io import BytesIO
import fitz  # PyMuPDF for PDF processing

# Load the tokenizer
tokenizer_file_path = "my_token.pk1"
with open(tokenizer_file_path, 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

# Load the pre-trained model
model = keras.models.load_model("my_model.h5")

# Streamlit app title and description
st.title("Fake News Detection")
st.write("Upload a file (PDF, text, articles, etc.) to check if it contains fake news or true information.")

# File upload widget
uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt"])

if uploaded_file is not None:
    # Read the file content
    if uploaded_file.type == "application/pdf":
        pdf_data = uploaded_file.read()
        pdf_text = ""
        pdf_document = fitz.open(stream=BytesIO(pdf_data))
        for page_num in range(len(pdf_document)):
            pdf_text += pdf_document[page_num].get_text()

        text = pdf_text
    else:
        text = uploaded_file.read().decode("utf-8")

    # Tokenize and pad the input text
    max_len = 128
    text = [str(text)]
    sequences = tokenizer.texts_to_sequences(text)
    padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

    # Make prediction
    prediction = model.predict(padded)

    # Define labels for classes (e.g., 0 for fake news, 1 for true news)
    class_labels = ["Fake News", "True News"]

    # Display the result
    st.write("### Prediction:")
    if prediction[0] >= 0.5:
        st.write(f"The uploaded content is **{class_labels[1]}** (Confidence: {prediction[0][0]:.2f})")
    else:
        st.write(f"The uploaded content is **{class_labels[0]}** (Confidence: {1 - prediction[0][0]:.2f})")

