import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
import pyautogui as pt

# Get the camera device
cap = cv2.VideoCapture(0)

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities


# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data')

# Actions that we try to detect
actions = np.array(['run', 'punch', 'placeblock'])

# Thirty videos worth of data
no_sequences = 30

# Videos are 30 frames in length
sequence_length = 30


def mediapipe_detection(image, model):
    # cv2 reads images in bgr so we need to convert it to rgb to process it in hollistic model
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Image no longer writeable
    image.flags.writeable = False

    # Make prediction
    results = model.process(image)
    
    # Set back to writeable
    image.flags.writeable = True

    # Convert it back to BGR format
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image, results

def draw_landmarks(image, results):

    # Apply drawing landmarks to the frame. Does not return the image
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))
    
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))

# Extracts landmark keypoints
def extract_keypoints(results):

    # Error handle if there are no landmarks in a hand just return an array of zeroes
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])
