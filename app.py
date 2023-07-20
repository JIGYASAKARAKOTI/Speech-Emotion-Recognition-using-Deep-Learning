import os
from flask import Flask, request, jsonify, render_template
import librosa
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the LSTM model
model = tf.keras.models.load_model('model.h5')

# Define the function to extract MFCC features
def extract_mfcc(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc


# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for handling favicon requests
@app.route('/favicon.ico')
def favicon():
    return '', 204

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Create the "uploads" directory if it doesn't exist
            if not os.path.exists('uploads'):
                os.makedirs('uploads')
                
            filepath = os.path.join('uploads', file.filename)
            file.save(filepath)
            mfcc_features = extract_mfcc(filepath)
            mfcc_features = np.expand_dims(mfcc_features, axis=0)
            mfcc_features = np.expand_dims(mfcc_features, axis=-1)
            predicted_probs = model.predict(mfcc_features)
            predicted_label = np.argmax(predicted_probs, axis=1)
            labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Pleasant Surprised', 'Sad']
            emotion = labels[predicted_label[0]]
            os.remove(filepath)  # Remove the uploaded file after prediction
            return jsonify({'emotion': emotion})
    return jsonify({'error': 'Please provide an audio file.'})

if __name__ == '__main__':
    app.run()
