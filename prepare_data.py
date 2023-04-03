
# Import and Install Dependencies
import cv2
import numpy as np
import os
import mediapipe as mp
from mediapipe_utils import *

def prepare_folders():
    for action in actions:
        for sequence in range(no_sequences):
            try:
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except: pass # If folders created just pass

# prepare_folders() # uncomment this out if you haven't created the MP_Data folder yet

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    # While web cam is opened

    for action in actions:
        # Loop through sequences (number of videos)
        for sequence in range(no_sequences):
            for frame_num in range(sequence_length):
                
                # Read feed. We get two values, a return value and our frame. 
                ret, frame = cap.read()

                # Make detection
                image, results = mediapipe_detection(frame, holistic)

                draw_styled_landmarks(image, results)

                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collection frames for {} Video Number {}'.format(action, sequence), (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    
                    # Take a break at the start of next video
                                   # Show frame to screen
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(2000)
                else:
                    cv2.putText(image, 'Collection frames for {} Video Number {}'.format(action, sequence), (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)

                # NEW Export keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break