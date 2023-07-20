# Speech-Emotion-Recognition-using-Deep-Learning
Jigyasa Karakoti
## Project Overview
Through all the available senses humans can actually sense the emotional state of their communication partner. Emotional detection is natural for humans but it is a very difficult task for computers; although they can easily understand content-based information, accessing the depth behind the content is difficult and that’s what speech emotion recognition (SER) sets out to do. It is a system through which various audio speech files are classified into different emotions such as happy, sad, anger and neutral by computer. SER can be used in areas such as the medical field or customer call centers. The aim of this project is to build a model that can predict emotion in a speech audio clip.
## Dataset
The TESS (Toronto emotional speech set data) Dataset from Kaggle contains 2800 audio file samples representing different emotions including angry, happy, sad, fearful, neutral, disgust, and surprise.
## Process
Data Loading and Exploration: The TESS dataset is loaded and explored to understand the distribution of different emotions.

Data Preprocessing: The audio data is preprocessed, and relevant features (Mel-frequency cepstral coefficients - MFCC) are extracted from the audio files to use as input for the LSTM model.

Model Architecture: The LSTM model is constructed with dropout layers to avoid overfitting. It is designed to take the MFCC features as input and predict the emotion class.

Model Training: The model is trained on the preprocessed data using categorical cross-entropy as the loss function and Adam optimizer.

Model Evaluation: The trained model's performance is evaluated on validation and testing datasets, and accuracy metrics are used to assess its performance.

Web Application: A Flask-based web application is developed to allow users to upload audio files and receive real-time predictions of the emotion in the audio clip using the trained model.
## Requirements
Jupyter Notebook,
Python 3.x,
Pandas,
NumPy,
Matplotlib,
Seaborn,
Librosa,
TensorFlow,
Keras,
Flask,
## Conclusion
Achieved an impressive 89.18% accuracy on the validation set and an overall accuracy of 89.18% on the testing data, demonstrating the model's effectiveness in emotion recognition.The successful implementation of this project opens up opportunities for practical applications in emotion recognition from speech, such as sentiment analysis, customer feedback analysis, and emotion-aware virtual assistants. Additionally, the project provides valuable experience in building end-to-end data science projects, from data exploration to deploying web applications.
