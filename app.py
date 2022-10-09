import cv2 as cv
import mediapipe as mp
import numpy as np
import time
from functions import VM


            
def main():
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
    