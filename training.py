import cv2 as cv
import mediapipe as mp
import numpy as np
import time
import os
from functions import VM



def collect_data(image, results):
    '''
    for a in fsl:
        for sequence in range(seq):
            for frame_num in range(seq_lenght):
                if frame_num == 0: 
                    cv.putText(image, 'STARTING COLLECTION', (120,200), 
                                cv.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv.LINE_AA)
                    cv.putText(image, 'Collecting frames for {} Video Number {}'.format(a, sequence), (15,12), 
                                cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)
                            # Show to screen
                    cv.imshow('OpenCV Feed', image)
                    cv.waitKey(1000)
                else: 
                    cv.putText(image, 'Collecting frames for {} Video Number {}'.format(a, sequence), (15,12), 
                                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)
                            # Show to screen
                    cv.imshow('OpenCV Feed', image)
                        
                        # NEW Export keypoints
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, a, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)

                    '''