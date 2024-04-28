import os
import pickle
import time

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import numpy as np

from train_classifier import train_classifier

def train_model():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, max_num_hands=2)

    DATA_DIR = './data'

    data = []
    labels = []
    for dir_ in os.listdir(DATA_DIR):
        
        for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
            data_aux = []

            x_ = []
            y_ = []

            actual_path = os.path.join(DATA_DIR, dir_, img_path)
            with open(actual_path, 'rb') as f:
                image_data = f.read()
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)


            # img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            print(dir_, "  ", img_path)
            if results.multi_hand_landmarks:
                print(dir_, " _______________________", img_path)
                for hand_landmarks in results.multi_hand_landmarks:

                    if(img_path=='0.jpg'):
                        mp_drawing.draw_landmarks(
                            img_rgb, # image to draw
                            hand_landmarks, # model output
                            mp_hands.HAND_CONNECTIONS, # hand connections
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())
                        cv2.imshow("img",img_rgb)
                        while(1) :
                            if cv2.waitKey(25) == ord('q'):
                                break

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        x_.append(x)
                        y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))
                    
                padding_req = 84 - len(data_aux)
                if(padding_req): 
                    data_aux = np.pad(data_aux, (0,padding_req), mode='constant')
                print(len(data_aux))

                data.append(data_aux)
                labels.append(dir_)

    print("harsh _________")
    print(labels)
    f = open('data.pickle', 'wb')
    pickle.dump({'data': data, 'labels': labels}, f)
    f.close()
    cv2.destroyAllWindows()
    time.sleep(3)
    train_classifier()
