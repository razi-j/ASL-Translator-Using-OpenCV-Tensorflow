import cv2 as cv
import mediapipe as mp
import numpy as np

mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils

def point_detection(image, model): #image from cv; holistic model from mp
    image = cv.cvtColor(cv.flip(image,1), cv.COLOR_BGR2RGB) #process color and flip
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv.cvtColor(cv.COLOR_RGB2BGR)
    return image, results

def draw_points(image, results):
    mp_draw.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_draw.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_draw.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
    mp_draw.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

def draw_styled_points(image, results):
    mp_draw.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                            mp_draw.DrawingSpec(color=(), thickness=1, circle_radius=1),#landmark color
                            mp_draw.DrawingSpec(color=(), thickness=1, circle_radius=1))#connection color
    
    mp_draw.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    
    mp_draw.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
    
    mp_draw.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

cap = cv.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence= 0.5, min_tracking_confidence= 0.5) as holistic:
    while cap.isOpened():
        ret, img = cap.read()
        # detects landmarks/points
        img, results = point_detection(img, holistic)
        # draws landmarks and connections
        draw_points(img, results)
        # displays camera feed with landmarks
        cv.imshow("VertoMotus", img)
        if cv.waitKey(10) == 27:
            break

    cap.release()
    cv.destroyAllWindows()