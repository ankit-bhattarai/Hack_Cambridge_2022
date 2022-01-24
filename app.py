import numpy as np
from PIL import ImageOps, Image
import tensorflow as tf

from keras.models import load_model
import streamlit as st

st.header("Farming and AI")
st.write("This web app uses two AI models which can be used in Corn Farming")

st.subheader("Healthy vs Infected")
st.write("This model determines whether a plant leaf is healthy or not")
st.write("The AI Model used was trained on images of corn leaf was obtained from:")
st.write("https://www.kaggle.com/qramkrishna/corn-leaf-infection-dataset")

st.subheader("Seedling Classification")
st.write("This model determines whether a seedling is a corn or weed. If it is a weed, it can also classify the type")
st.write("The data used to train ths model was obtained from:")
st.write("https://www.kaggle.com/vbookshelf/v2-plant-seedlings-dataset")

st.subheader("Image Classification")
option = st.radio("Please select which AI Model you would like to use:",
                  ("Classify Healthy vs Infected Leaves", "Determine Seedling Type"))

display_scores = st.checkbox("Display the prediction scores from the model")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

corn_disease_model = load_model("corn_leaf_disease_model")
corn_disease_model_labels = ['Healthy corn', 'Infected']

seedling_model = load_model("seedling_detection_model")
seedling_model_labels = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed',
                         'Common wheat', 'Fat Hen', 'Loose Silky-bent', 'Maize', 'Scentless Mayweed',
                         'Shepherd\'s Purse', 'Small-flowered Cranesbill', 'Sugar beet']


def process_img(img_data):
    """Processes the raw image into a numpy array"""
    size = (180, 180)
    image = ImageOps.fit(img_data, size, Image.ANTIALIAS)
    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)
    return img_array


def calculate_scores(img_data, model):
    """Calculates the scores of the image against the model"""
    processed_image = process_img(img_data)
    predictions = model.predict(processed_image)
    score = tf.nn.softmax(predictions[0])
    return np.array(score)


def display_results(score, labels):
    """Displays the results of the model prediction"""
    st.write("This image is most likely {} with a {:.2f} % confidence.".format(labels[np.argmax(score)],
                                                                              100 * np.max(score)))


def display_stats(score, labels):
    """Displays exact scores of the prediction for each category"""
    st.subheader("Scores")
    dictionary = dict(zip(labels, score))
    sorted_dict = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)
    for key, value in sorted_dict:
        st.write("{}: {:.1e}".format(key, value))


def results(image):
    if option == "Classify Healthy vs Infected Leaves":
        model, labels = (corn_disease_model, corn_disease_model_labels)
    else:
        model, labels = (seedling_model, seedling_model_labels)
    score = calculate_scores(image, model)
    display_results(score, labels)
    if display_scores:
        display_stats(score, labels)




def main():
    if file is None:
        st.text("Please upload an image file")
    else:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        results(image)


if __name__ == "__main__":
    main()
