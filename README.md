# Mood Detection Using Facial Expressions

This project implements an emotion detection system that classifies facial expressions using Convolutional Neural Networks (CNNs). The system supports both real-time emotion detection through a webcam and image-based detection by uploading static images. The model was trained on the FER-2013 dataset and deployed using Python libraries such as TensorFlow, OpenCV, and Streamlit.

## Project Structure

```plaintext
.
├── image_detection_app.py                # Static image-based emotion detection
├── model_summary.txt                     # Summary of the CNN model architecture
├── preprocessing_training_data.ipynb      # Data preprocessing, model building, training, and saving model architecture/weights
├── real_time_detection_app.py             # Real-time emotion detection via webcam
├── requirements.txt                      # Required Python libraries for the project

```

## Features

- Real-time Emotion Detection: Detects emotions through a webcam in real-time.
- Static Image Detection: Upload images to classify emotions based on facial expressions.
- CNN Model: Built using a multi-layered CNN architecture.
- Model Training: Code provided to train the model on the FER-2013 dataset and save the model architecture (.json) and weights (.h5).

## How to run
1. **Clone the Repository:**
```bash
git clone https://github.com/shivanshi1/Mood-detection-using-facial-expressions.git
```

2. **Create and Activate a Virtual Environment:**

- **Create a virtual environment:**
```bash
python -m venv venv
```
- **Activate the virtual environment:**
Windows:
```bash
venv\Scripts\activate
```
3. **Install the Required Libraries:** Install the libraries from the requirements.txt file.
```bash
pip install -r requirements.txt
```
4. **Train the Model:**
If you don't have the pre-trained .json and .h5 files for the model architecture and weights, you can generate them by running the preprocessing_training_data.ipynb notebook. This will train the model on the FER-2013 dataset and save the files.

5. **Run the Applications:**

- **For real-time emotion detection**: This will open a web browser where emotions will be detected in real-time through your webcam.
```bash
streamlit run real_time_detection_app.py
```

- **For static image detection**: This will open a web browser where you can upload images, and the system will detect emotions from facial expressions.
```bash
streamlit run image_detection_app.py
```
