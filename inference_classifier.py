import pickle

import cv2
import mediapipe as mp
import numpy as np
from PIL import ImageFont, ImageDraw, Image

def start_inference():
    default_clear = "default_clear"
    default_ok = "default_ok"
    default_list = [default_clear, default_ok]
    global sentence
    sentence  = ""
    my_dict = {}

    def clear_sentence():
        global sentence
        sentence = ""
        my_dict.clear()


    def add_to_sentence(s):
        global sentence
        if(s not in default_list):
            sentence+= " "+s
        my_dict.clear()


    font_path = "Akshar_Unicode.ttf"
    font_size = 30
    font = ImageFont.truetype(font_path, font_size)
    width, height = 600, 150

    def add_senctence_to_frame(frame, text):
        pil_image = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil_image)
        text_width, text_height = draw.textsize(text, font=font)
        text_position = ((frame.shape[1] - text_width) // 2, (frame.shape[0] - text_height) // 2)
        draw.text(text_position, text, font=font, fill=(255, 255, 255))
        frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']

    cap = cv2.VideoCapture(0)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    labels_dict = {0: 'A', 1: 'B', 2: default_ok}
    max_key = "default_ok"
    while True:

        data_aux = []
        x_ = []
        y_ = []
        predicted = ""

        ret, frame = cap.read()
        # frame = cv2.imread("0.jpg")

        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            for hand_landmarks in results.multi_hand_landmarks:
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

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            padding_req = 84 - len(data_aux)
            if(padding_req): 
                data_aux = np.pad(data_aux, (0,padding_req), mode='constant')
            
            
            prediction = model.predict([np.asarray(data_aux)])
            predicted = (prediction[0])

            print(predicted)

            if predicted in my_dict: 
                my_dict[predicted]+=1
            else:
                my_dict[predicted]=0

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            # cv2.putText(frame, predicted + " "+str(my_dict[predicted]), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3,
            #             cv2.LINE_AA)
            
            max_key = max(my_dict, key=my_dict.get)
            if (predicted == default_ok and my_dict[predicted]>5):
                print("___________________here")
                add_to_sentence(max_key)
            elif (predicted == default_clear and my_dict[predicted]>5):
                clear_sentence()
            elif(my_dict[predicted]>50):
                my_dict.clear()
            
            
        # if(max_key not in default_list):
        #     cv2.putText(frame, max_key + "  "+ str(my_dict[max_key]), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
        #         cv2.LINE_AA)
        
        # text = np.ones((100, 1200, 3), dtype=np.uint8) * 255
        # cv2.putText(text, sentence, (10, text.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
        #                 cv2.LINE_AA)
        
        cv2.imshow('frame', frame)


        
        blank_image = np.ones((height, width, 3), np.uint8) * 255

    # Convert the blank image to PIL format
        pil_image = Image.fromarray(blank_image)
        draw = ImageDraw.Draw(pil_image)

        # print(sentence)

    # Write Hindi text on the image
        text = ""
        if(max_key not in default_list): text = max_key
        text_width, text_height = draw.textsize(text, font=font)
        text_position = (1, 1)
        draw.text(text_position, text, font=font, fill=(0, 0, 0))

        text = ""
        if(predicted not in default_list): text = predicted 
        text_width, text_height = draw.textsize(text, font=font)
        text_position = ((width - text_width) // 2, (height - text_height) // 2)
        draw.text(text_position, text, font=font, fill=(0, 0, 0))

        text = sentence
        text_width, text_height = draw.textsize(text, font=font)
        text_position = ((1), (height - text_height-1))
        draw.text(text_position, text, font=font, fill=(0, 0, 0))

        cv_image = np.array(pil_image)

        cv2.imshow('Hindi Text', cv_image)




        cv2.waitKey(1)
        if cv2.waitKey(25) == ord('q'): break


    cap.release()
    cv2.destroyAllWindows()

# start_inference()