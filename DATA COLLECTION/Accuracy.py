import os
import pandas as pd
import numpy as np
def Averages(folder):
    os.chdir(f"./DATA COLLECTION/{folder}/Samples/")
    df = pd.read_csv("averages.csv")
    mean = df["Averages"].mean()
    with open("/home/andy/VertoMotus-FSLTranslator-Using-OpenCV-Tensorflow/DATA COLLECTION/Accuracy.txt", "a") as f:
        f.write(f"{str(mean)} - {folder}\n")
Averages("thank you")