import pandas as pd

# Append labels to formatted data
def appendLabelsToData(dataFile, labelFile, outputFile):
    data = pd.read_csv(dataFile)
    labels = pd.read_csv(labelFile, header=None, names=['class'])

    if len(data) != len(labels):
            print(f"LENGTHS DONT MATCH !! data: {len(data)}, labels: {len(labels)}")

    labels.columns = ['class']
    data['class'] = labels['class'].values
    data.to_csv(outputFile, index=False)

def rm_main(pfeatureResultFilePath, labelsFilePath, outputFilePath):
    appendLabelsToData(pfeatureResultFilePath, labelsFilePath, outputFilePath)

if __name__ == "__main__":
    # # Glue labels back on
    # # PLEASE put pfeature training file after making it into number in the unfinished folder
    #appendLabelsToData("../Dataset/Unfinished/pfeature_result_normalized.csv", "../DataSet/Unfinished/trainingLabels.csv", "../Dataset/full_normalized.csv")
    appendLabelsToData("../Dataset/Unfinished/pfeature_result.csv", "../DataSet/Unfinished/trainingLabels.csv", "../Dataset/full_not_normalized.csv")