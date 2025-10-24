import pandas as pd
import numpy as np

def reducelabelRows(input, output, label, percent):
    data = pd.read_csv(input)
    labelCol = data.columns[-1]
    
    labelRows = data[data[labelCol] == label]
    numToRemove = int(len(labelRows) * percent)
    
    if numToRemove > 0:
        indicesToRemove = np.random.choice(labelRows.index, size=numToRemove, replace=False)
        data = data.drop(index=indicesToRemove)
    
    data.to_csv(output, index=False)
    print(f"Removed {numToRemove} rows of {label}")

if __name__ == "__main__":
    # Fix up params n allat
    input = "trainingRelabeled.csv" # We need to organize these files somehow this is whack
    output = "trainingUndersampled.csv"
    label = "nonDRNA"
    percent = 0.2 # 20%
    
    reducelabelRows(input, output, label, percent)