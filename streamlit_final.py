import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import requests
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from skimage import io, color
from skimage.feature import graycomatrix, graycoprops
from scipy import fftpack

st.set_page_config(
    page_title="Clasificación de imágenes",
    page_icon="iconoweb.ico",
    layout="wide"
)

st.markdown("""
    <style>
        .main .block-container {
            transform: scale(0.8);
            transform-origin: top center;
        }
    </style>
    """, unsafe_allow_html=True)

model_options = {
    "DenseNet121": "best_model-DenseNet121.h5",
    "DenseNet201": "best_model-DenseNet201.h5",
    "EfficientNetB7": "best_model-EfficientNetB7.h5",
    "ResNet151v2": "best_model-ResNet151v2.h5",
    "VGG19": "best_model-VGG19.h5"
}

category_names = {
    0: "Buildings",
    1: "Forest",
    2: "Glacier",
    3: "Mountain",
    4: "Sea",
    5: "Street"
}

def load_selected_model(selected_model_path):
    return tf.keras.models.load_model(selected_model_path, compile=False)

def preprocess_image(image, model_name):
    preprocess_input = None
    model_mapping = {
        "DenseNet": tf.keras.applications.densenet.preprocess_input,
        "EfficientNet": tf.keras.applications.efficientnet.preprocess_input,
        "ResNet": tf.keras.applications.resnet_v2.preprocess_input,
        "VGG": tf.keras.applications.vgg19.preprocess_input
    }
    for key, value in model_mapping.items():
        if model_name.startswith(key):
            preprocess_input = value
            break
    image = image.resize((150, 150))
    image = np.expand_dims(img_to_array(image), axis=0)
    if preprocess_input:
        image = preprocess_input(image.copy())
    return image

def load_image_from_url(url, selected_model_name):
    response = requests.get(url)
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        processed_image = preprocess_image(image, selected_model_name)
        return image, processed_image
    else:
        st.sidebar.error(f"Unable to load image from URL. Status code: {response.status_code}")
        return None, None

def get_prediction_info(model, processed_image, selected_model_name, image_name):
    if processed_image is not None:
        class_prediction = model.predict(processed_image)
        class_index = np.argmax(class_prediction)
        class_name = category_names.get(class_index, "Custom classification")
        probability_values = class_prediction[0] * 100
        return {
            "Model": selected_model_name,
            "Prediction": class_name,
            "Probability (%)": probability_values[class_index],
            "Image Name": image_name
        }
    return None

def export_to_csv(predictions):
    df = pd.DataFrame(predictions)
    df.to_csv("predictions.csv", index=False)
    st.sidebar.success("CSV file exported successfully as 'predictions.csv'")

def perform_probability_chart(model, processed_image):
    class_prediction = model.predict(processed_image)
    probability_values = class_prediction[0] * 100
    category_labels = list(category_names.values())
    st.subheader("Probability by class:")
    chart_data = {category_labels[i]: probability_values[i] for i in range(len(category_labels))}
    st.bar_chart(chart_data, height=250)

def perform_classification(model, processed_image, selected_model_name):
    if processed_image is not None:
        class_prediction = model.predict(processed_image)
        class_name = category_names.get(np.argmax(class_prediction), "Custom classification")
        st.subheader("Result:")
        st.write("Model:")
        st.markdown(f"**{selected_model_name}**")
        st.write("Prediction:")
        st.markdown(f'<p style="font-size:30px; text-align:center;"><strong>{class_name}</strong></p>', unsafe_allow_html=True)

def display_results(image, processed_image, selected_model, selected_model_name, image_name=None):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(image, caption=image_name, use_column_width=True, width=300)
    if selected_model_name != "NA":
        with col2:
            perform_classification(selected_model, processed_image, selected_model_name)
        with col3:
            perform_probability_chart(selected_model, processed_image)

def process_image(image_path):
    with Image.open(image_path) as img:
        img = img.resize((50, 50))
        img = np.array(img)
    return img.reshape((img.shape[0] * img.shape[1], 3))

def show_eda(train_dir, test_dir, categories):
    dataset_type = st.selectbox("**Select the type of dataset:**", ["Train", "Test"])
    if dataset_type == "Train":
        data_dir = train_dir
        imagen_suffix = "_train"
    else:
        data_dir = test_dir
        imagen_suffix = "_test"
    counts = {}
    for category in categories:
        path = os.path.join(data_dir, category)
        if os.path.exists(path):
            image_files = os.listdir(path)
            counts[category] = len(image_files)
    col1, col2 = st.columns(2)
    with col1:
        plt.figure(figsize=(30, 20))
        plt.bar(counts.keys(), counts.values(), color='blue')       
        plt.title('Image count per category', fontsize=60)
        plt.xticks(rotation=45)
        plt.tick_params(axis='x', labelsize=40)
        plt.tick_params(axis='y', labelsize=30)
        st.pyplot(plt)
    with col2:
        st.image(f"colors{imagen_suffix}.jpg", width=800)
    st.image(f"texture{imagen_suffix}.jpg", width=1500)
    conclusion_html = """
    <h2><b>Conclusion:</b></h2>
    <p>The data exploration conducted shows a notable uniformity in the quantity of photos per category, which is essential to avoid biases in machine learning models based on these images. Although a rich variety is observed in the image bank, the absence of the color green is surprising, especially considering the diversity of natural landscapes that could include it. This omission could influence the interpretation and analysis of data related to natural areas. Additionally, the image set exhibits an interesting contrast and uniformity in colors, reflecting the quality and diversity of the image bank. This characteristic is crucial for training robust and accurate models in image classification and recognition.</p>
    """
    st.markdown(conclusion_html, unsafe_allow_html=True)

def show_project_info():
    st.markdown("<div style='text-align: center; font-size: 50px; font-weight: bold;'>Images classification project</div>", unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)
    image_path_1 = "siro.jpg"
    image_path_2 = "fran.jpg"
    col1, col2 = st.columns(2)
    with col1:
        st.image(image_path_1, caption="Siro Matsubara", width=300)
    with col2:
        st.image(image_path_2, caption="Fran Jimena", width=300)
    st.write("""
    <h2><b>Technical Description:</b></h2>
    
    The "Image Classification using Deep Learning Models" project focuses on employing machine learning techniques and classification models to analyze and categorize landscape images into different classes such as buildings, forests, glaciers, mountains, seas, and streets. The primary aim is to achieve high accuracy in landscape classification using deep learning models.
    
    <h3><b>Technologies Used:</b></h3>
    
    TensorFlow and Keras: TensorFlow library with the Keras API was used to develop and train deep learning models. Keras simplifies the creation and training of neural network models.
    
    Pretrained Models: Pretrained models such as DenseNet121, DenseNet201, EfficientNetB7, ResNet151v2, and VGG19 were leveraged. These models were previously trained on massive datasets and used as a base for transfer learning.
    
    Image Preprocessing: Landscape images were preprocessed to ensure compatibility with the models. This included resizing images to a specific size and normalizing pixel intensities.
    
    Feature Extraction: Pretrained models were used to extract significant features from images, aiding in meaningful image representation.
    
    Evaluation and Validation: A labeled dataset was used to train and validate the models. Evaluation metrics like accuracy were employed to measure model performance.
    
    <h3><b>Results:</b></h3>
    
    The project achieved a high level of success in landscape classification, with accuracy reaching up to 0.91 in the ResNet152V2 model. This signifies the model's ability to accurately identify the correct category of an image, demonstrating the effectiveness of deep learning models in landscape image classification tasks.
    
    <h3><b>Potential Applications:</b></h3>
    
    This project holds potential applications in various domains such as automatic identification of landscapes in images, organizing large sets of images by category, and automation of tasks related to landscape image identification and labeling.
    
    In summary, the "Image Classification using Deep Learning Models" project employs machine learning techniques and classification models to achieve high accuracy in categorizing landscape images, holding significant applications in image processing and visual data organization.
    """, unsafe_allow_html=True)

def image_classification_page():
    st.sidebar.header("Load Images")
    selected_option = st.sidebar.radio("Select how to load the images:", ("Upload files", "Use URL"), key="image_load_option")
    selected_model_name = st.sidebar.selectbox("Select the model:", list(model_options.keys()), key="model_selection")
    selected_model_path = model_options[selected_model_name]
    selected_model = load_selected_model(selected_model_path)
    predictions = []
    st.markdown(
        f"""
        <div style='text-align: center;'>
            <h1>Image Classification</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<br><br>", unsafe_allow_html=True)
    if selected_option == "Upload files":
        uploaded_images = st.sidebar.file_uploader("Upload images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
        if uploaded_images is not None:
            for uploaded_image in uploaded_images:
                image = Image.open(uploaded_image)
                processed_image = preprocess_image(image, selected_model_name)
                prediction = get_prediction_info(selected_model, processed_image, selected_model_name, uploaded_image.name)
                if prediction:
                    predictions.append(prediction)
                    display_results(image, processed_image, selected_model, selected_model_name, image_name=uploaded_image.name)
    else:
        image_url = st.sidebar.text_input("Enter the image URL:")
        if image_url:
            image, processed_image = load_image_from_url(image_url, selected_model_name)
            if image is not None and processed_image is not None:
                display_results(image, processed_image, selected_model, selected_model_name)
    if predictions:
        export_button = st.sidebar.button("Export to CSV")
        if export_button:
            export_to_csv(predictions)
    else:
        st.sidebar.info("Make predictions before exporting the CSV.")

def main():
    menu_options = {
        "📄 Project Information": "information",
        "📊 EDA": "eda",
        "🖼️ Image Classification": "clasification"
    }
    st.sidebar.title("Menu")
    selected_option = st.sidebar.radio(
        "Select:",
        list(menu_options.keys())
    )
    if selected_option == "📄 Project Information":
        show_project_info()
    elif selected_option == "📊 EDA":
        show_eda("data/train", "data/test", ["buildings", "forest", "glacier", "mountain", "sea", "street"])
    elif selected_option == "🖼️ Image Classification":
        image_classification_page()

if __name__ == "__main__":
    main()