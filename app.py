import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load the trained generator model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("generator_model.h5")

generator = load_model()

# Streamlit UI
st.title("Handwritten Digit Generator")
digit = st.number_input("Enter a digit (0–9)", min_value=0, max_value=9, step=1)

if st.button("Generate"):
    num_images = 5
    noise = tf.random.normal([num_images, 100])
    labels = tf.keras.utils.to_categorical([digit] * num_images, 10)

    generated_images = generator.predict([noise, labels])
    generated_images = (generated_images + 1) / 2.0

    # Display images
    st.subheader(f"Generated images of digit {digit}")
    fig, axs = plt.subplots(1, num_images, figsize=(10, 2))
    for i in range(num_images):
        axs[i].imshow(generated_images[i, :, :, 0], cmap='gray')
        axs[i].axis('off')
    st.pyplot(fig)
