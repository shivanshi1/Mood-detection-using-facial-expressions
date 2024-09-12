import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
import streamlit as st
from PIL import Image

# Load the model from JSON file
with open('models/emotion_model.json', 'r') as json_file:
    loaded_model_json = json_file.read()

model = model_from_json(loaded_model_json)

# Load weights into the model
model.load_weights('models/emotion_model.weights.h5')

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Function to detect emotion in an image
def detect_emotion_image(model, image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face.astype('float32') / 255.0
        face = np.reshape(face, (1, 48, 48, 1))
        prediction = model.predict(face)
        emotion = emotion_labels[np.argmax(prediction)]
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(image, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    return image

# Streamlit web app
def main():
    st.title("Facial Expression Recognition on Images")
    st.write("Upload an image, and the model will identify facial expressions in the image.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Convert the uploaded file to an OpenCV image
        image = np.array(Image.open(uploaded_file))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert from RGB (PIL) to BGR (OpenCV)

        # Run emotion detection
        processed_image = detect_emotion_image(model, image)

        # Convert back to RGB for displaying in Streamlit
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        
        # Display the processed image
        st.image(processed_image, caption="Processed Image", use_column_width=True)

if __name__ == '__main__':
    main()
