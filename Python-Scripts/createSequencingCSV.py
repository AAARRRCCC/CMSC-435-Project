import pandas as pd
import os


def createSequencingCSV(dataFrame: pd.DataFrame, sequenceOutput):
    with open(sequenceOutput, "w", encoding="utf-8") as f:
        for i, (_, row) in enumerate(dataFrame.iterrows(), start=0):
            rowData = " ".join(str(val) for val in row.values)
            
            f.write(f">seq_{i}\n")
            f.write(f"{rowData}\n")
def rm_main(testSetPath= "C:/Users/rbrad/Desktop/VCU Fall 2025/CMSC 435/CMSC-435-Project/sequences_test.txt"):

    df= pd.read_csv(testSetPath, header=None)
    print(f"✓ [createSequencingCSV] Read {len(df)} sequences from {os.path.basename(testSetPath)}")
    createSequencingCSV(df, "C:/Users/rbrad/Desktop/VCU Fall 2025/CMSC 435/CMSC-435-Project/Dataset/Unfinished/pfeatureSequenced.fa")
    print(f"✓ [createSequencingCSV] Wrote {len(df)} sequences to FASTA format")
    return "junk"