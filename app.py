import cv2 as cv
import time
from functions import VM
import numpy as np
import os 
import tensorflow as tf



def main():
    # Needed Variables for Detection
    sequence =[]
    threshold = 0.9
    sentence =[]
    predictions = []
    signs = sorted(os.listdir("Keypoint_Data/"))
    # Initialization of TFLite Model
    interpreter = tf.lite.Interpreter(model_path="VertoMotus.tflite")
    interpreter.allocate_tensors()
    input_d = interpreter.get_input_details()
    output_d = interpreter.get_output_details()
    
    print(signs)
    cap = cv.VideoCapture(0)
    with VM.mp_holistic.Holistic(min_detection_confidence= 0.5, min_tracking_confidence= 0.5) as holistic:
        # Main Loop

        while cap.isOpened():
            start = time.time()
            ret, img = cap.read()
            # detects landmarks/points
            img, results = VM.point_detection(img, holistic)
            # draws landmarks and connections
            VM.draw_styled_points(img, results)
            # displays camera feed with landmarks
            
            keypoints = VM.extract_keypoints(results) # Extract Keypoints to be used to predict actions
            sequence.append(keypoints) # Append Keypoints to Variable
            sequence = sequence[-30:]


            try:
                if results.right_hand_landmarks or results.left_hand_landmarks:
                    if len(sequence) == 30:
                        interpreter.set_tensor(input_d[0]["index"], np.array(np.expand_dims(sequence,axis=0), dtype=np.float32))
                        interpreter.invoke()
                        output_data = interpreter.get_tensor(output_d[0]["index"])
                        pred  = np.squeeze(output_data)
                        
                        #res = np.argmax(pred)
                        predictions.append(np.argmax(pred))
                        if np.unique(predictions[-10:])[0] == np.argmax(pred):
                            if pred[np.argmax(pred)] > threshold:
                                print(pred, pred[np.argmax(pred)],signs[np.argmax(pred)])
         
                else: pass
            except: pass

            end = time.time()
            fps = 1/(end-start)
            #
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
    