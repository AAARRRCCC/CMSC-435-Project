import pandas as pd
from pandas import DataFrame as df

# Get random split depending on trainingPercent
def splitData(fileName, trainingOutput, testingOutput, trainingPercent):
    fullData = pd.read_csv(fileName, header=None)
    trainingData = fullData.sample(frac=trainingPercent)
    testingData = fullData.drop(trainingData.index)

    trainingData.to_csv(trainingOutput, index=False, header=False)
    testingData.to_csv(testingOutput, index=False, header=False)

    return trainingData, testingData

# Take raw sequences and labels and split it 
def removeLabels(filePath, labelOutput):
    dataFrame = pd.read_csv(filePath)

    labels = dataFrame[[dataFrame.columns[1]]].copy()
    labels.to_csv(labelOutput, index=False, header=False)

    data = dataFrame.drop(dataFrame.columns[1], axis=1)
    return data, labels

# Make sequencing files
def createSequencingCSV(dataFrame: df, sequenceOutput):
    with open(sequenceOutput, "w", encoding="utf-8") as f:
        for i, (_, row) in enumerate(dataFrame.iterrows(), start=1):
            rowData = " ".join(str(val) for val in row.values)
            
            f.write(f">seq_{i}\n")
            f.write(f"{rowData}\n")

# Append labels to formatted data
def appendLabelsToData(dataFile, labelFile, outputFile):
    data = pd.read_csv(dataFile)
    labels = pd.read_csv(labelFile, header=None, names=['class'])

    if len(data) != len(labels):
            print(f"LENGTHS DONT MATCH !! data: {len(data)}, labels: {len(labels)}")

    # fixed that off by one error that bc the labels didnt have a header

    labels.columns = ['class']
    data['class'] = labels['class'].values
    data.to_csv(outputFile, index=False)

if __name__ == "__main__":
    # Remove labels
    #unlabeledTrainingData, _ = removeLabels("Dataset/StartingData.txt", "DataSet/Unfinished/trainingLabels.csv")

    # Create sequencing file
    #createSequencingCSV(unlabeledTrainingData, "DataSet/Unfinished/RYAN_FILE.fa")

    # # Glue labels back on
    # # PLEASE put pfeature training file after making it into number in the unfinished folder
    #appendLabelsToData("../Dataset/Unfinished/pfeature_result_normalized.csv", "../DataSet/Unfinished/trainingLabels.csv", "../Dataset/full_normalized.csv")
    appendLabelsToData("../Dataset/Unfinished/pfeature_result.csv", "../DataSet/Unfinished/trainingLabels.csv", "../Dataset/full_not_normalized.csv")

    pass