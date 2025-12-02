import pandas as pd

# Append labels to formatted data
def appendLabelsToData(dataFile, labelFile):
    outputFile = labelFile.replace("trainingLabels.csv", "labeledData.csv")
    data = dataFile
    labels = pd.read_csv(labelFile, header=None, names=['class'])

    if len(data) != len(labels):
            print(f"LENGTHS DONT MATCH !! data: {len(data)}, labels: {len(labels)}")

    labels.columns = ['class']
    data['class'] = labels['class'].values
    data.to_csv(outputFile, index=False)
    return data

def rm_main(data_df, labelsFilePath):
    df = appendLabelsToData(data_df, labelsFilePath)
    df.to_csv("AAC-DPC-PAAC-RRI-ACR-PCP-AAIAfterPfeature.csv",index=False)
    return df
if __name__ == "__main__":
    # # Glue labels back on
    # # PLEASE put pfeature training file after making it into number in the unfinished folder
    #appendLabelsToData("../Dataset/Unfinished/pfeature_result_normalized.csv", "../DataSet/Unfinished/trainingLabels.csv", "../Dataset/full_normalized.csv")
    appendLabelsToData("../Dataset/Unfinished/pfeature_result.csv", "../DataSet/Unfinished/trainingLabels.csv", "../Dataset/full_not_normalized.csv")