import streamlit as st
import tensorflow as tf 
import os
import time
from model.model import load_model
from utils.helper_functions import load_frames_and_annotation, num_to_char

st.set_page_config(page_title="AI Visual Speech Recognition",layout="wide")

with st.sidebar: 
    st.write("")
    st.write("")
    st.write("")
    st.image("assets/sidebar_image.png")
    st.info("Code: [GitHub Link](.)")
    st.info("Dataset: [Link](https://spandh.dcs.shef.ac.uk//gridcorpus/)")
    st.info("Aknowledgements: [Link](https://arxiv.org/abs/1611.01599)")
    
st.title("AI Visual Speech Recognition")

st.write("The Visual Speech Recognition AI, also known as Lip Reading AI, recognizes and interprets what an individual is saying by only analyzing the visual information obtained from their facial movements, effectively transcribing what the individual is saying without relying on the audio.")

st.write("The videos used for both training and testing the model are a sample taken from the GRID corpus dataset, which consists of audio and video recordings of 1000 sentences spoken by each of 34 speakers. The sentence structure follows a specific format:")
st.warning("put red at G9 now")
st.write("Link to GRID Corpus: https://spandh.dcs.shef.ac.uk//gridcorpus/")

st.divider()

st.subheader("Test and experiment with the model")

st.write("Choose a video from the dropdown below:")
videos = os.listdir(os.path.join("..", "data", "videos"))
selected_option = st.selectbox("", videos, label_visibility="collapsed")

col1, col2, col3 = st.columns([0.2,0.6,0.2])

with col2:
    # Convert .mpg file to .mp4 file
    file_path = os.path.join('..','data','videos', selected_option)
    os.system(f"ffmpeg -i {file_path} -vcodec libx264 selected_video.mp4 -y")
            
    # Read and display video
    video_file = open("selected_video.mp4", "rb") 
    video_bytes = video_file.read() 
    st.video(video_bytes)

st.write("The visual speech recognition by the AI:")

model = load_model()
frames, annotation = load_frames_and_annotation(tf.convert_to_tensor(file_path))
prediction = model.predict(tf.expand_dims(frames, axis=0))
decode = tf.keras.backend.ctc_decode(prediction, [75], greedy=True)[0][0].numpy()
decoded_prediction = tf.strings.reduce_join(num_to_char(decode)).numpy().decode("utf-8")

with st.spinner("Model is analyzing..."):
    time.sleep(3)
st.success(decoded_prediction)
st.write("Remember the model does not use any audio and only relies on the visuals of the video.")