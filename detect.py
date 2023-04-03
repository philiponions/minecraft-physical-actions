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

def gui_action(action, prev_action):
    print(action)
    if action == "run" and not prev_action == "run":
        print("holding w")
        pt.keyDown('w')
    elif action == "punch" and not prev_action == 'punch':
        print("holding click")
        pt.mouseDown()
    elif action == "placeblock" and not prev_action == "placeblock":
        print("right click")
        pt.rightClick()
    elif action == "standby" and prev_action == "placeblock":
        print("releasing w")
        pt.keyUp("w")
    elif action == "standby" and prev_action == "punch":
        print("releasing click")
        pt.mouseUp(button='left')
    # elif action == "standby":
    #     print("Standing by")
    #     print("the previous action was", prev_action)

    return action 

prev_action = "standby"

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

        action = prev_action
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0), verbose = 0)[0]
            # print(actions[np.argmax(res)])
            if res[np.argmax(res)] > threshold:
                action = actions[np.argmax(res)]
                
            else: 
                action = "standby"

            # prev_action = gui_action(action, prev_action)
            #     cv2.putText(image, actions[np.argmax(res)], (15, 50),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
                
            #     prev_action = gui_action(actions[np.argmax(res)], prev_action)

            # else: 
            #     cv2.putText(image, "standby", (15, 50),
            #             cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.putText(image, action, (15, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)

        # prev_action = gui_action(action, prev_action)
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()