import os
import pandas as pd
def Averages(folder):
    data = []
    os.chdir(f"./DATA COLLECTION/{folder}/Samples/")
    num = 0
    print(os.listdir())
    for i in os.listdir():
        if i.endswith(".csv"):
            df=pd.read_csv(i, header=None)
    #print(df.to_string())
            for j in df[0]:
                if j == folder:
                    num +=1
            data.append((num/df[0].count())*100)
            print(i, (num/df[0].count())*100)
            num=0
    df = pd.DataFrame(data, columns=["Averages"])
    df.to_csv("averages.csv", index=False)
Averages("thank you")
