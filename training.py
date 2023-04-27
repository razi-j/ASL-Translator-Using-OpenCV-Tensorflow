import cv2 as cv
import mediapipe as mp
import numpy as np
import time
import os
from functions import VM



def collect_data():
    cap = cv.VideoCapture(0)
    with VM.mp_holistic.Holistic(min_detection_confidence= 0.5, min_tracking_confidence= 0.5) as holistic:
        while cap.isOpened():
            for a in VM.fsl:
                for sequence in range(VM.seq):
                    for frame_num in range(VM.seq_lenght):
                        
                        start = time.time()
                        ret, img = cap.read()
                        # detects landmarks/points
                        img, results = VM.point_detection(img, holistic)
                        # draws landmarks and connections
                        VM.draw_styled_points(img, results)
                        # displays camera feed with landmarks
                        img = cv.flip(img,1)

                        if frame_num == 0: 
                            cv.putText(img, 'STARTING COLLECTION', (120,200), 
                                        cv.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv.LINE_AA)
                            cv.putText(img, 'Collecting frames for {} Video Number {}'.format(a, sequence), (15,12), 
                                        cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)
                                    # Show to screen
                            cv.imshow('VertoMotus', img)
                            cv.waitKey(3000)
                        else: 
                            cv.putText(img, 'Collecting frames for {} Video Number {}'.format(a, sequence), (15,12), 
                                            cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)
                                    # Show to screen
                            cv.imshow('VertoMotus', img)
                                
                                # NEW Export keypoints
                        keypoints = VM.extract_keypoints(results)
                        npy_path = os.path.join(VM.DATA_PATH, a, str(sequence), str(frame_num))
                        np.save(npy_path, keypoints)

                        end = time.time()
                        fps = 1/(end-start)
                        
                        
                        cv.putText(img,f'FPS: {int(fps)}', (20, 70), cv.FONT_HERSHEY_PLAIN, 1.5, (0,255,0), 2)
                        cv.imshow("VertoMotus", img)


                        key = cv.waitKey(10)
                        if key == 27: #escape
                            break
            cap.release()
            cv.destroyAllWindows()

if __name__ == "__main__":
    VM.build_folder()
    collect_data()