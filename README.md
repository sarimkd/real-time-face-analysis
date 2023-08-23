 # Real-Time Face Analysis using Deepface

Welcome to the Real-Time Face Analysis project! This application leverages the power of computer vision and deep learning to detect emotions in real-time using webcam input. The project is built using Python, OpenCV, TensorFlow, Keras, and the DeepFace library. It's a great example of how modern technologies can be combined to create an interactive and engaging application.

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Getting Started](#getting-started)
    - [Installation](#installation)
    - [Usage](#usage)
4. [How It Works](#how-it-works)
    - [Emotion Classification](#emotion-classification)
5. [About the Developer](#about-the-developer)
6. [Contributing](#contributing)
7. [License](#license)

## Introduction
Real-Time Face Analysis is a project that demonstrates the capabilities of modern computer vision and deep learning techniques. By analyzing live webcam input, the application can identify emotions such as happiness, sadness, anger, and more.

The project is developed using Python and utilizes popular libraries such as OpenCV for video capture and manipulation, TensorFlow and Keras for deep learning model development, and the DeepFace library for emotion classification. The application is user-friendly and provides an intuitive way to interact with the emotion detection model.

## Features
- Real-time emotion detection from webcam input.
- Display of detected emotions on the video stream.
- Simple and easy-to-use Streamlit user interface.
- Integration of deep learning using TensorFlow and Keras.
- Utilization of DeepFace for accurate emotion classification.

## Getting Started
### Installation
1. Clone this repository to your local machine:
```
git clone https://github.com/sarimkd/real-time-face-analysis.git
cd real-time-face-analysis
```
2. Set up a virtual environment (recommended):
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install the required dependencies:
```
pip install -r requirements.txt
```
### Usage
1. Run the application using Streamlit:
```
streamlit run app.py
```
2. The application will launch in your default web browser. You'll be presented with options to start the webcam feed and see the real-time emotion detection in action.

## How It Works
### Emotion Classification
Deep learning comes into play here. The project utilizes a pre-trained custom CNN model developed using TensorFlow and Keras. The model is loaded and used to classify the extracted face regions into various emotions such as happy, sad, surprised, etc. The predicted emotion is then displayed on the video stream.

## About the Developer
This project was developed by Sarim Khan, a passionate machine learning engineer with a strong interest in reinforcement learning, machine learning, and game development. Connect with me on [LinkedIn](https://www.linkedin.com/in/sarimkhanskd/) to stay updated on my latest projects and insights.

## Contributing
Contributions to this project are welcome! If you find any issues or want to add new features, feel free to fork the repository and submit a pull request. 

In my professional experience working with Deepface, in addition to Emotion Detection, I have implemented ethnicity, age and gender detection with Deepface. Feel free to add these to this project!

## License
This project is licensed under the MIT License - see the LICENSE file for details.

Enjoy exploring the world of real-time emotion detection! Your feedback and suggestions are greatly appreciated. Feel free to contact me for any inquiries or collaboration opportunities.

