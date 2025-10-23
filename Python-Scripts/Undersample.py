import pandas as pd
import numpy as np

def reduceClassRows(input_csv, output_csv, class_name, percent):
    data = pd.read_csv(input_csv)
    class_col = data.columns[-1]
    
    class_rows = data[data[class_col] == class_name]
    n_remove = int(len(class_rows) * percent)
    
    if n_remove > 0:
        remove_indices = np.random.choice(class_rows.index, size=n_remove, replace=False)
        data = data.drop(index=remove_indices)
    
    data.to_csv(output_csv, index=False)
    print(f"Removed {n_remove} rows of {class_name}")

if __name__ == "__main__":
    # Fix up params n shi
    input_csv = "trainingRelabeled.csv" # We need to organize these files somehow this is whack
    output_csv = "trainingUndersampled.csv"
    class_name = "nonDRNA"
    percent = 0.2 # 20%
    
    reduceClassRows(input_csv, output_csv, class_name, percent)