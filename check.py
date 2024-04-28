import os
import time

import cv2
import mediapipe as mp


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)


DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

dataset_size = 10

# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Error: Could not open camera.")



# for j in range(number_of_classes):
while 1:

    # j = input("Please enter word: ")
    # if(j=="q"): 
    #     break

    # if not os.path.exists(os.path.join(DATA_DIR, str(j))):
    #     os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}')

    done = False
    while True:
        # ret, frame = cap.read()
        frame = cv2.imread("0.jpg")

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                mp_drawing.draw_landmarks(
                    img, # image to draw
                    hand_landmarks, # model output
                    mp_hands.HAND_CONNECTIONS, # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                cv2.imshow("img",img)
                # while(1) :
                #     if cv2.waitKey(25) == ord('q'):
                #         break



        cv2.putText(frame, 'Ready... Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)

        if cv2.waitKey(25) == ord('q'):
            break
    

cap.release()
cv2.destroyAllWindows()
