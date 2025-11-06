import pandas as pd
from sklearn.preprocessing import StandardScaler
import sys

def normalize(input_path):
    df = pd.read_csv('../Dataset/Unfinished/pfeature_result.csv')
    #df = pd.read_csv('../Dataset/Unfinished/pfeature_with_labels.csv')


    # using Z-score normalization because the interwebs said its better for ML and showing the differences more clearly
    scaler = StandardScaler()
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    df.to_csv('../Dataset/Unfinished/pfeature_result_normalized.csv', index=False)
    #df.to_csv('../Dataset/finished/pfeature_with_labels_normalized.csv', index=False)

    return df

def rm_main(input_path):
    df = pd.read_csv('../Dataset/Unfinished/pfeature_result.csv')
    #df = pd.read_csv('../Dataset/Unfinished/pfeature_with_labels.csv')


    # using Z-score normalization because the interwebs said its better for ML and showing the differences more clearly
    scaler = StandardScaler()
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    df.to_csv('../Dataset/Unfinished/pfeature_result_normalized.csv', index=False)
    #df.to_csv('../Dataset/finished/pfeature_with_labels_normalized.csv', index=False)

    return df


if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        normalize(input_path)
    else:
        print("Please provide the input file path as a command-line argument.")