import pandas as pd

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
def createSequencingCSV(dataFrame: pd.DataFrame, sequenceOutput):
    with open(sequenceOutput, "w", encoding="utf-8") as f:
        for i, (_, row) in enumerate(dataFrame.iterrows(), start=1):
            rowData = " ".join(str(val) for val in row.values)
            
            f.write(f">seq_{i}\n")
            f.write(f"{rowData}\n")

def rm_main(dataFilePath, labelsOutputFilePath, pfeatureOutputFilePath):
    unlabeledTrainingData, _ = removeLabels(dataFilePath, labelsOutputFilePath)
    createSequencingCSV(unlabeledTrainingData, pfeatureOutputFilePath)

if __name__ == "__main__":
    # Remove labels
    unlabeledTrainingData, _ = removeLabels("Dataset/StartingData.txt", "DataSet/Unfinished/trainingLabels.csv")

    # Create sequencing file
    createSequencingCSV(unlabeledTrainingData, "DataSet/Unfinished/RYAN_FILE.fa")