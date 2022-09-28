import cv2 as cv
import mediapipe as mp
import numpy as np
import time
import os

mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils


Data_Path = os.path.join("./Data")

asl = np.array([]) #things to put here: asl words, phrases
seq = 30 # number of videos to be used for data collection
seq_lenght = 30 # number of frames to be used

def folderBuild():
    for a in asl:
        for sequences in range(seq):
            try:
                os.makedirs(os.path.join(Data_Path, asl, str(sequences)))
            except:
                pass


def point_detection(image, model): #image from cv; holistic model from mp
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB) #process color and flip
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    return image, results

def draw_points(image, results):
    mp_draw.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_draw.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_draw.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
    mp_draw.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

def draw_styled_points(image, results):
    mp_draw.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                            mp_draw.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=3),#landmark color
                            mp_draw.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=1))#connection color

    mp_draw.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                            mp_draw.DrawingSpec(color=(84, 44, 44), thickness=2, circle_radius=3),
                            mp_draw.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=1))
    '''
    mp_draw.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                            mp_draw.DrawingSpec(color=(255,170,170), thickness=1, circle_radius=1),
                            mp_draw.DrawingSpec(color=(255,255,255), thickness=1, circle_radius=1))
    
    mp_draw.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(42,43,42), thickness=2, circle_radius=3),
                        mp_draw.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=1))
    '''
def extract_keypoints(results):
    # list comprehension to loop over results and get needed data, then arranged to np.array. flattened to turn it into one array. else is to make a placeholder
    pose = np.array([[res.x,res.y,res.z,res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x,res.y,res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x,res.y,res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    face = np.array([[res.x,res.y,res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    
    # puts all data into one array
    a = np.concatenate([lh, rh, pose, face])
    with open("hakdog.csv", "w") as f:
        np.savetxt(f, a, delimiter=',')

     

def collect_data():
    pass

def main():
    cap = cv.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence= 0.5, min_tracking_confidence= 0.5) as holistic:
        while cap.isOpened():
            start = time.time()
            ret, img = cap.read()
            # detects landmarks/points
            img, results = point_detection(img, holistic)
            # draws landmarks and connections
            draw_styled_points(img, results)
            # displays camera feed with landmarks
            
            end = time.time()
            fps = 1/(end-start)
            
            img = cv.flip(img,1)
            cv.putText(img,f'FPS: {int(fps)}', (20, 70), cv.FONT_HERSHEY_PLAIN, 1.5, (0,255,0))
            extract_keypoints(results)
            cv.imshow("VertoMotus", img)
            
            if cv.waitKey(10) == 27:
                break
            elif cv.waitKey(10) =="t":
                collect_data()

        cap.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()
