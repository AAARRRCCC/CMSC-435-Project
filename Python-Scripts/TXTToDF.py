import pandas as pd

def rm_main(filepath = "C:/Users/rbrad/Desktop/VCU Fall 2025/CMSC 435/CMSC-435-Project/Dataset/StartingData.txt"):
    df = pd.read_csv(filepath)
    return df