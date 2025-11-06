import os
import pandas as pd

def rm_main(data):
    path = "../TempfilesAndOutput/predictions_output.csv"
    header = not os.path.exists(path)
    data.to_csv(path, mode='a', header=header, index=False)
    return data