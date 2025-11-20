import subprocess
import os
import pandas as pd

def rm_main(pfeatureinputfilepath):
    output_dir = os.path.dirname(pfeatureinputfilepath)
    pfeatureoutputfilepath = output_dir + "/pfeature_result.csv"
    
    subprocess.run("python pfeature_comp.py -i ../Dataset/Unfinished/pfeatureSequenced.fa -o CSVTEMP.csv -j TPC")
    df = pd.read_csv("CSVTEMP.csv")
    return df

if __name__ == "__main__":
    input_path = "..Dataset/Unfinished/pfeatureSequenced.fa"
    rm_main(input_path)