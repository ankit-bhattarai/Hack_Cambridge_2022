import numpy as np
from PIL import ImageOps, Image
import tensorflow as tf

from keras.models import load_model

model = load_model("model_corn_disease_selection")

import streamlit as st

st.write("Determine Whether a plant leaf is healthy or not")
st.write("The AI Model used was trained on images of corn leaf taken from this website:")
st.write("https://www.kaggle.com/qramkrishna/corn-leaf-infection-dataset")
st.write("This model works best with corn leaf images.")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

labels = ['Healthy corn', 'Infected']


def processed_img(img_data):
    size = (180, 180)
    image = ImageOps.fit(img_data, size, Image.ANTIALIAS)
    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    return "Conclusion: This image is most likely {} with a {:.2f} percent confidence.".format(labels[np.argmax(score)],
                                                                                               100 * np.max(score))


def main():
    if file is None:
        st.text("Please upload an image file")
    else:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        st.write(processed_img(image))


if __name__ == "__main__":
    main()
