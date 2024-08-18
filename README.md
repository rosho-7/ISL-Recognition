
# ISL Recognition Project

This project focuses on the recognition of Indian Sign Language (ISL) using machine learning and deep learning techniques. The dataset used for this project can be found at [Mendeley Data](https://data.mendeley.com/datasets/kcmpdxky7p/1).

## Table of Contents

- [Dataset](#dataset)
- [Notebooks](#notebooks)
  - [word_level.ipynb](#word_levelipynb)
  - [Continued training.ipynb](#continued-trainingipynb)
- [Streamlit App](#streamlit-app)
  - [Project_Overview.py](#project_overviewpy)
  - [01ISL-Translator.py](#01isl-translatorpy)
  - [02Metrics.py](#02metricspy)
- [Saved Models](#saved-models)
- [Setup](#setup)
- [License](#license)

## Dataset

The dataset used in this project is provided by Mendeley Data and can be accessed at the following link: [Indian Sign Language Dataset](https://data.mendeley.com/datasets/kcmpdxky7p/1).

The dataset consists of images and videos of various sign language gestures used in Indian Sign Language.

## Notebooks

### word_level.ipynb

This notebook focuses on word-level recognition in ISL. It includes the following sections:

1. **Data Loading**: Loading the dataset for word-level recognition.
2. **Feature Extraction**: Extracting features from the dataset.
3. **Model Building**: Building the machine learning model for word-level recognition.
4. **Training**: Training the model with the extracted features.
5. **Evaluation**: Evaluating the model's performance on the test set.
6. **Results Visualization**: Visualizing the results of the model.

### Continued training.ipynb

This notebook focuses on the continued training of a pre-trained model on the ISL dataset -word_level.ipynb. It includes the following sections:

1. **Data Preprocessing**: Loading and preprocessing the dataset for training.
2. **Model Architecture**: Defining the architecture of the deep learning model.
3. **Training**: Training the model again with the augmented dataset .
4. **Evaluation**: Evaluating the model's performance on the test set.
5. **Fine-Tuning**: Fine-tuning the model for improved accuracy.



## Streamlit App

![WhatsApp Image 2024-07-30 at 11 56 13_c285ae04](https://github.com/user-attachments/assets/13494ead-8a67-48fa-8ee1-e4664dc073d5)


### Project_Overview.py

This script provides an overview of the ISL recognition project. It includes sections on the project's motivation, dataset, and model architecture.

### 01ISL-Translator.py

This script implements the ISL Translator application. It allows users to upload images or videos of ISL gestures and get predictions for the corresponding signs. The application uses a pre-trained model to make predictions.

### 02Metrics.py

This script provides metrics and visualizations for the ISL recognition model. It includes sections for evaluating the model's performance and visualizing various metrics.

## Saved Models

The following models are trained models on the dataset:

![image](https://github.com/user-attachments/assets/2e60aeed-7e36-4b83-9617-2739270152e0)


- `model2_updated_again0.h5`
- `model2_updated_again1.h5`
- `model2_updated_again2.h5`
- `model2_updated_again3.h5`
- `model2_updated_again4.h5`

These models are saved in the HDF5 format and can be loaded for making predictions in the Streamlit app or for further training.

## Setup

To run the notebooks and Streamlit app, you need to have the following dependencies installed:

- Python 3.x
- Jupyter Notebook
- TensorFlow
- Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Streamlit

You can install the dependencies using the following command:

```bash
pip install tensorflow keras opencv-python numpy pandas matplotlib scikit-learn streamlit
```

## Usage

1. Clone the repository:

```bash
git clone <repository-url>
cd <repository-directory>
```

2. Download the dataset from [Mendeley Data](https://data.mendeley.com/datasets/kcmpdxky7p/1) and place it in the `data` directory.

3. Open the notebooks using Jupyter Notebook:

```bash
jupyter notebook
```

4. To run the Streamlit app, use the following command:

```bash
streamlit run <script_name>.py
```

Replace `<script_name>` with the appropriate script name (`Project_Overview.py`, `01ISL-Translator.py`, or `02Metrics.py`).


## License

This project is licensed under the MIT License.

---

Feel free to modify this README file as per your specific project requirements.
