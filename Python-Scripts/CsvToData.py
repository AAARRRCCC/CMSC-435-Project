import pandas as pd

def rm_main(filepath):
    df = pd.read_csv(filepath)
    return df