import os
import pandas as pd

def rm_main(data):
    path = "../TempfilesAndOutput/predictions_output.csv"
    header = not os.path.exists(path)
    data.to_csv(path, mode='w', header=header, index=False)
    data = data['prediction(class)']
    data.to_csv("../TempfilesAndOutput/Team1.csv", mode='w', header=False, index = False)
    return data
    