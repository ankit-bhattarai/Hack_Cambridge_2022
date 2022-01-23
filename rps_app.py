from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import ImageOps, Image
import pickle
import cv2
import tensorflow as tf

from keras.models import load_model
model = load_model("model_corn_disease_selection")

import streamlit as st

st.write("Determine Whether a plant leaf is healthy or not:")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

labels = ['Healthy corn', 'Infected']


def processed_img(img_data):
    data = np.ndarray(shape=(1, 180, 180, 3), dtype=np.float32)
    size = (180, 180)
    image = ImageOps.fit(img_data, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    data[0] = normalized_image_array
    answer = model.predict(data)
    print(answer)
    score = tf.nn.softmax(answer[0])
    print(score)
    txt = "{}".format(labels[np.argmax(score)])
    print(txt)
    return txt


def main():
    if file is None:
        st.text("Please upload an image file")
    else:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        prediction = processed_img(image)

        st.write("Conclusion: It is", prediction)


if __name__ == "__main__":
    main()
