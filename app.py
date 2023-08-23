import numpy as np
import cv2
import streamlit as st
from tensorflow import keras
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode

# load model
emotion_dict = {0:'angry', 1 :'happy', 2: 'neutral', 3:'sad', 4: 'surprise'}
# load json and create model
json_file = open('emotion_model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)

# load weights into new model
classifier.load_weights("emotion_model1.h5")

#load face
try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class Faceemotion(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        #image gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(
                x + w, y + h), color=(255, 0, 0), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_dict[maxindex]
                output = str(finalout)
            label_position = (x, y)
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

def main():
    # Face Analysis Application #
    linkedin = "https://www.linkedin.com/in/sarimkhanskd"
    st.title("Real Time Face Analysis")
    activities = ["Home", "About"]
    choice = st.sidebar.selectbox("Choose an option", activities)
    st.sidebar.write("Developed by [Sarim Khan](%s)" % linkedin)
    st.sidebar.write("Email : sarimkhanskd@gmail.com")
    
    if choice == "Home":
        st.subheader("Emotion Detection")
        st.write("This is a Face Emotion detection application using OpenCV, Custom CNN model and Streamlit.") 
        st.write("This is a module from a Voice of Customer System implemented for a multinational retailer. The entire system included ethnicity, age and gender detection as well.\nThis module only detects faces and emotions in real time using web cam feed.")
        st.write("Click on start to use webcam. Your data won't be stored because I care about your security and can't afford the storage.")
        webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
                        video_processor_factory=Faceemotion)

    elif choice == "About":
        st.subheader("About this app")
        st.write("Real time face emotion detection application using OpenCV, Custom Trained CNN model and Streamlit.")
        st.write("This Application is developed by Sarim Khan using Streamlit Framework, OpenCV, Tensorflow and Keras libraries. If you're on LinkedIn and want to connect, just click my name on the sidebar and shoot me a request. If you have any suggestion or just to want to let me know how much my work sucks, email me at sarimkhanskd@gmail.com.")
        st.write("Thanks for Visiting")

    else:
        pass


if __name__ == "__main__":
    main()