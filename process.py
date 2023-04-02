''' 
ACTION RECOGNITION  

Author: Nicholas Renotte
Video: https://www.youtube.com/watch?v=doDUihpj6ro&t=7438s

'''

# Import and Install Dependencies
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
import pyautogui

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

sequence = []
sentence = []
threshold = 0.4

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation="relu", input_shape=(30,126)))
model.add(LSTM(128, return_sequences=True, activation="relu"))
model.add(LSTM(64, return_sequences=False, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(actions.shape[0], activation="softmax"))

model.load_weights('action.h5')


def gui_action(action):
    if action == "run":
        pyautogui.keyDown('w')
    if action == "punch":
        pyautogui.click()
    if action == "placeblock":
        pyautogui.rightClick()

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    # While web cam is opened
    while cap.isOpened():
        # Read feed. We get two values, a return value and our frame. 
        ret, frame = cap.read()

        # Make detection
        image, results = mediapipe_detection(frame, holistic)

        draw_styled_landmarks(image, results)

        keypoints = extract_keypoints(results)
        sequence.insert(0, keypoints)
        sequence = sequence[:30]

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            # print(actions[np.argmax(res)])
            if res[np.argmax(res)] > threshold:
                cv2.putText(image, actions[np.argmax(res)], (15, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
                
                gui_action(actions[np.argmax(res)])

        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#     # While web cam is opened

#     for action in actions:
#         # Loop through sequences (number of videos)
#         for sequence in range(no_sequences):
#             for frame_num in range(sequence_length):
                
#                 # Read feed. We get two values, a return value and our frame. 
#                 ret, frame = cap.read()

#                 # Make detection
#                 image, results = mediapipe_detection(frame, holistic)

#                 draw_styled_landmarks(image, results)

#                 if frame_num == 0:
#                     cv2.putText(image, 'STARTING COLLECTION', (120, 200),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
#                     cv2.putText(image, 'Collection frames for {} Video Number {}'.format(action, sequence), (15, 12),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    
#                     # Take a break at the start of next video
#                                    # Show frame to screen
#                     cv2.imshow('OpenCV Feed', image)
#                     cv2.waitKey(2000)
#                 else:
#                     cv2.putText(image, 'Collection frames for {} Video Number {}'.format(action, sequence), (15, 12),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
#                     cv2.imshow('OpenCV Feed', image)

#                 # NEW Export keypoints
#                 keypoints = extract_keypoints(results)
#                 npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
#                 np.save(npy_path, keypoints)

 

#                 # Break gracefully
#                 if cv2.waitKey(10) & 0xFF == ord('q'):
#                     break



# for action in actions:
#     for sequence in range(no_sequences):
#         try:
#             os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
#         except: pass # If folders created just pass

cap.release()
cv2.destroyAllWindows()