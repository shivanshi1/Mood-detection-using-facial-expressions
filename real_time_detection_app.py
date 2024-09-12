import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
import streamlit as st

# Load the model from JSON file
with open('models/emotion_model.json', 'r') as json_file:
    loaded_model_json = json_file.read()

model = model_from_json(loaded_model_json)

# Load weights into the model
model.load_weights('models/emotion_model.weights.h5')

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Function to detect emotion in real-time and display in Streamlit
def detect_emotion_video(model):
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    stop_button = st.button('Stop')
    
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret or stop_button:
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Load the Haar cascade for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Extract the face
            face = gray[y:y+h, x:x+w]
            # Resize the face to 48x48 pixels
            face = cv2.resize(face, (48, 48))
            face = face.astype('float32') / 255.0
            face = np.reshape(face, (1, 48, 48, 1))
            # Predict the emotion
            prediction = model.predict(face)
            emotion = emotion_labels[np.argmax(prediction)]
            # Draw a rectangle around the face and put the emotion label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        # Display the resulting frame in Streamlit
        stframe.image(frame, channels="BGR")
    
    # Release the capture when done
    cap.release()

# Streamlit web app
def main():
    st.title("Real-time Emotion Detection")
    st.write("Click the 'Stop' button to stop the webcam.")
    detect_emotion_video(model)

if __name__ == "__main__":
    main()
