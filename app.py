import cv2 as cv
import time
from functions import VM
import numpy as np
import os 
from tensorflow.keras.models import load_model
import tensorflow as tf
            
def main():
    sequence =[]
    threshold = 0.4
    sentence =[]
    signs = os.listdir("Model_Data/")


    cap = cv.VideoCapture(0)
    with VM.mp_holistic.Holistic(min_detection_confidence= 0.5, min_tracking_confidence= 0.5) as holistic:
        while cap.isOpened():
            start = time.time()
            ret, img = cap.read()
            # detects landmarks/points
            img, results = VM.point_detection(img, holistic)
            # draws landmarks and connections
            VM.draw_styled_points(img, results)
            # displays camera feed with landmarks
            
            interpreter = tf.lite.Interpreter(model_path="LiteML.tflite")

            keypoints = VM.extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            #if len(sequence) == 30:
            #    res = model.predict(np.expand_dims(sequence, axis=0))[0]
            #   print(signs[np.argmax(res)])

            end = time.time()
            fps = 1/(end-start)
            
            img = cv.flip(img,1)
            cv.putText(img,f'FPS: {int(fps)}', (20, 70), cv.FONT_HERSHEY_PLAIN, 1.5, (0,255,0), 2)
            cv.imshow("VertoMotus", img)
            
            key = cv.waitKey(10)
            if key == 27: #escape
                break
            

        cap.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()
    