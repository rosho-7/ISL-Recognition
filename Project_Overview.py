import streamlit as st

# Introduction Section
st.markdown("""
### Introduction

This project presents an innovative approach to Indian Sign Language (ISL) recognition, tailored to accommodate the diverse regional languages of India. Leveraging a Convolutional Neural Network (CNN)-based architecture, the system aims to interpret and translate ISL gestures accurately across various linguistic contexts, addressing the unique challenges posed by India's linguistic diversity.
""")

# Methodology Section
st.markdown("""
### Methodology

The proposed system employs a robust CNN model to process image sequences of sign language gestures, extracting spatial and temporal features for precise gesture classification. The architecture integrates several key components:

1. **Image Preprocessing and Feature Extraction**:
   - **MediaPipe**: Utilized for hand keypoint detection, capturing detailed hand movements.
   - **MobileNetV2**: Employed for high-level feature extraction, focusing on intricate gesture details.

2. **Dimensionality Reduction**:
   - **Principal Component Analysis (PCA)**: Applied to reduce the dimensionality of the extracted features, ensuring efficient processing.

3. **Gesture Recognition**:
   - **Long Short-Term Memory (LSTM) Networks**: Used to recognize gestures by analyzing the temporal sequences of the extracted features.

**Diagram illustrating the methodology:**
""")

methodology_image_path = r"arch.png"  # Update this path to your image file
st.image(methodology_image_path, caption="Architecture of Model",width=500)

# Experimental Results Section
st.markdown("""
### Experimental Results

The system demonstrates high accuracy and efficiency in word-level gesture recognition, achieving 80-85% accuracy. However, it exhibits only mediocre accuracy in sentence-level gesture recognition, underscoring the complexity of contextual interpretation in sign language.
""")

# Significance and Future Directions Section
st.markdown("""
### Significance and Future Directions

Despite the challenges, this research underscores the potential of CNN-based architectures in bridging communication gaps for the deaf and hard-of-hearing communities in multilingual settings. The study highlights several future directions to enhance the system:

1. **Expanding the Training Dataset**:
   - Including diverse ISL gestures from various regions and dialects.
   - Collaborating with linguistic experts and the deaf community for comprehensive gesture coverage.

2. **Model Optimization**:
   - Reducing computational requirements to enhance real-time processing capabilities.

3. **Integrating Multimodal Inputs**:
   - Incorporating facial expressions and body movements to improve recognition accuracy.

4. **Developing Cross-Platform Solutions**:
   - Creating mobile applications and wearable devices for wider accessibility.
""")

# Conclusion Section
st.markdown("""
### Conclusion

This project marks a significant step towards creating inclusive communication tools that respect and reflect India's linguistic diversity. By addressing the outlined future directions, the system aims to build on its current foundation, making substantial strides towards a universally accessible and highly accurate ISL recognition and translation system.
""")
