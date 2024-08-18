import os
import numpy as np
import pandas as pd
import cv2
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNetV2
import mediapipe as mp
from googletrans import Translator
import streamlit as st
from PIL import Image

# Set page configuration
st.set_page_config(page_title="Sign Language Translator", layout="wide")

# Load and preprocess the DataFrame
@st.cache_data
def load_data():
    df = pd.read_csv("E:/augmented_word_csv.csv").dropna()
    new_base_path = r"D:/kcmpdxky7p-1/ISL_CSLRT_Corpus/"
    df["Frames path"] = df["Frames path"].apply(lambda x: new_base_path + x if not x.startswith("E:") else x)
    return df

df = load_data()
unique_sentences = df['Word'].unique()

# Load scaler, PCA, and models
@st.cache_resource
def load_resources():
    scaler = joblib.load('scaler.pkl')
    pca = joblib.load('pca.pkl')
    n_models = 5
    model_dir = 'models_ensemble'
    models = [load_model(os.path.join(model_dir, f'model2_updated_again{i}.h5')) for i in range(n_models)]
    cnn_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    return scaler, pca, models, cnn_model

scaler, pca, models, cnn_model = load_resources()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Initialize Google Translator
translator = Translator()

# Function to extract keypoints using MediaPipe
def extract_mediapipe_keypoints(image):
    image_uint8 = (image * 255).astype('uint8')  # Convert float32 to uint8
    results = hands.process(cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR))
    if results.multi_hand_landmarks:
        keypoints = []
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                keypoints.append((lm.x, lm.y, lm.z))
        keypoints = np.array(keypoints).flatten()
        if len(keypoints) < 63:
            keypoints = np.pad(keypoints, (0, 63 - len(keypoints)), mode='constant')
        return keypoints
    else:
        return np.zeros(63)

# Function to extract CNN features
def extract_cnn_features(image_array):
    image_array = np.expand_dims(image_array, axis=0)
    features = cnn_model.predict(image_array)
    return features.flatten()

# Function to combine keypoints and CNN features
def combine_features(image_array):
    keypoints = extract_mediapipe_keypoints(image_array)
    cnn_features = extract_cnn_features(image_array)
    combined_features = np.concatenate((keypoints, cnn_features))
    return combined_features

# Function to preprocess a single image
def preprocess_image(image_path, target_size=(224, 224)):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image.astype('float32') / 255.0
    return image

# Function to make predictions using the ensemble of models
def predict_with_ensemble(models, X):
    num_classes = models[0].output_shape[-1]
    predictions = np.zeros((X.shape[0], num_classes))

    for model in models:
        pred = model.predict(X)
        predictions += pred
    
    predictions /= len(models)  # Average the predictions
    return predictions

# Function for real-time prediction
def real_time_prediction(image_paths):
    processed_images = [preprocess_image(img_path) for img_path in image_paths]
    combined_features_list = [combine_features(img) for img in processed_images]
    combined_features_array = np.array(combined_features_list)
    combined_features_array_2d = combined_features_array.reshape(combined_features_array.shape[0], -1)
    combined_features_array_scaled = scaler.transform(combined_features_array_2d)
    reduced_features_array = pca.transform(combined_features_array_scaled)
    reduced_features_array = reduced_features_array.reshape((reduced_features_array.shape[0], 1, reduced_features_array.shape[1]))
    ensemble_predictions = predict_with_ensemble(models, reduced_features_array)
    
    return ensemble_predictions

# Function to translate predicted words
def translate_word(word, target_language):
    translated = translator.translate(word, dest=target_language)
    return translated.text

# Available languages for translation
available_languages = {
    'Spanish': 'es', 'French': 'fr', 'German': 'de', 'Hindi': 'hi',
    'Chinese (Simplified)': 'zh-cn', 'Japanese': 'ja', 'Korean': 'ko',
    'Russian': 'ru', 'Italian': 'it', 'Portuguese': 'pt', 'Arabic': 'ar',
    'Bengali': 'bn', 'Tamil': 'ta', 'Telugu': 'te', 'Marathi': 'mr',
    'Gujarati': 'gu', 'Urdu': 'ur', 'Kannada': 'kn', 'Malayalam': 'ml',
    'Odia': 'or', 'Punjabi': 'pa', 'Assamese': 'as', 'Maithili': 'mai',
    'Bodo': 'brx', 'Nepali': 'ne', 'Sindhi': 'sd', 'Kashmiri': 'ks',
    'Manipuri': 'mni', 'Sanskrit': 'sa', 'Tulu': 'tvl', 'Dogri': 'doi',
    'Santali': 'sat', 'Meitei': 'mni', 'Konkani': 'kok', 'Kannada': 'kn', 
    'Telugu': 'te'
}

st.title("Indian Sign Language Translator")

# File uploader selection
uploaded_files = st.file_uploader("Upload Images", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

language_options = ["-- Select language --"] + list(available_languages.keys())
selected_language = st.selectbox("Select target language", options=language_options)

if st.button("Predict"):
    if uploaded_files and selected_language != "-- Select language --":
        image_paths = []
        for uploaded_file in uploaded_files:
            img = Image.open(uploaded_file)
            img_array = np.array(img)
            img_path = f"temp_{uploaded_file.name}"
            cv2.imwrite(img_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
            image_paths.append(img_path)

        predictions = real_time_prediction(image_paths)
        target_language_code = available_languages[selected_language]

        for i, prediction in enumerate(predictions):
            predicted_label = np.argmax(prediction)
            predicted_word = unique_sentences[predicted_label]
            confidence = prediction[predicted_label]
            translated_word = translate_word(predicted_word, target_language_code)
            
            st.write(f"*Prediction for image {i+1}:* {predicted_word} (Confidence: {confidence:.2f})")
            st.write(f"*Translation in {selected_language}:* {translated_word}")
            st.image(uploaded_files[i], caption=f"Image {i+1}")

    else:
        st.warning("Please upload images and select a target language.")
