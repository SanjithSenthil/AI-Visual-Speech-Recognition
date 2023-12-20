import tensorflow as tf
import cv2
from typing import List
import os 

# Define character set
character_set = [x for x in "abcdefghijklmnopqrstuvwxyz123456789'?! "]

# Convert character to number: 'a' = 1, 'b' = 2, ..., 'z' = 26
char_to_num = tf.keras.layers.StringLookup(vocabulary=character_set, oov_token="")

# Convert number to character: 1 = 'a', 2 = 'b', ..., 26 = 'z'
num_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)

# Return frames and annotation of the video
def load_frames_and_annotation(path): 
    path = bytes.decode(path.numpy())
    file_name = path.split('/')[-1].split('.')[0]
    video_path = os.path.join('..','data','videos',f'{file_name}.mpg')
    annotation_path = os.path.join('..','data','annotations',f'{file_name}.align')
    # Get normalized frames of the video
    cap = cv2.VideoCapture(video_path)
    frames = []
    for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))): 
        ret, frame = cap.read()
        frame = tf.image.rgb_to_grayscale(frame)
        frames.append(frame[210:256,115:255,:])
    cap.release()
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    frames = tf.cast((frames - mean), tf.float32) / std
    # Get annotation of the video
    with open(annotation_path, 'r') as f: 
        lines = f.readlines() 
    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != 'sil': 
            tokens = [*tokens,' ',line[2]]
    annotation = char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]
    return frames, annotation