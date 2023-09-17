import cv2
import mediapipe as mp
import numpy as np

# Define your CNN model and characters list here
# cnn = ...
# characters = [...]

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:

    res, frame = cap.read()

    if not res:  
        print("Error reading frame from webcam")
        break

    x, y, c = frame.shape

    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame_resized = cv2.resize(framergb, (128, 128))
    frame_resized = frame_resized / 255.0
    frame_input = np.expand_dims(frame_resized, axis=0)

    prediction = cnn.predict(frame_input)
    classID = np.argmax(prediction)
    className = characters[classID]  
    print(className)


    cv2.putText(frame, className, (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 
                2, cv2.LINE_AA)


    cv2.imshow("Output", frame)

    if cv2.waitKey(1) == ord('q'):
        break
