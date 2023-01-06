import cv2 as cv
import mediapipe as mp
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import pyttsx3

class VM: 
    mp_holistic = mp.solutions.holistic
    mp_draw = mp.solutions.drawing_utils
    DATA_PATH = os.path.join("./Keypoint_Data")

    fsl = np.array(["i'm sorry"]) #things to put here: asl words, phrases
    seq = 50 # number of videos to be used for data collection
    seq_lenght = 30 # number of frames to be used per video
    

    def build_folder():
        for a in VM.fsl:
            for sequences in range(VM.seq):

                try:
                    os.makedirs(os.path.join(VM.DATA_PATH, a, str(sequences)))
                except:
                    pass

    def point_detection(image, model): #image from cv; holistic model from mp
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB) #process color and flip
        image.flags.writeable = False
        results = model.process(image)
        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        return image, results

    def draw_styled_points(image, results):
        VM.mp_draw.draw_landmarks(image, results.right_hand_landmarks, VM.mp_holistic.HAND_CONNECTIONS,
                                VM.mp_draw.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=3),#landmark color
                                VM.mp_draw.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=1))#connection color

        VM.mp_draw.draw_landmarks(image, results.left_hand_landmarks, VM.mp_holistic.HAND_CONNECTIONS,
                                VM.mp_draw.DrawingSpec(color=(84, 44, 44), thickness=2, circle_radius=3),
                                VM.mp_draw.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=1))
        
        VM.mp_draw.draw_landmarks(image, results.face_landmarks, VM.mp_holistic.FACEMESH_TESSELATION,
                                VM.mp_draw.DrawingSpec(color=(255,170,170), thickness=1, circle_radius=1),
                                VM.mp_draw.DrawingSpec(color=(255,255,255), thickness=1, circle_radius=1))

    def extract_keypoints(results):
        # list comprehension to loop over results and get needed data, then arranged to np.array. flattened to turn it into one array. else is to make a placeholder
        lh = np.array([[res.x,res.y,res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x,res.y,res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        face = np.array([[res.x,res.y,res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        
        # puts all data into one array
        a = np.concatenate([face, lh, rh])

        lm_list = []

        for lm in a:
            base = a[0]
            lm_list.append(lm - base) 
        lm_list = np.array(lm_list, dtype=np.float32)

        return a

    def convert():
        #  Load Model
        model = load_model("./VertoMotus_MLmodel3.h5")
        #  Initialize TFLite Converter
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        #   Enable flags
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
            tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
        ]
        #  Convert Model to TFLite Model
        tfLite_Model = converter.convert()
        #   Write Model to tflite file
        with open("./VertoMotus2.tflite","wb") as f:
            f.write(tfLite_Model)
    def tts(text):
        vclient = pyttsx3.init()
        vclient.say(text)
        vclient.runAndWait()

if __name__ == "__main__":
    VM.convert()
