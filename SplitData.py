import pandas as pd
from pandas import DataFrame as df

#Get random split depending on trainingPercent
def splitData(fileName, trainingOutput, testingOutput, trainingPercent):
    fullData = pd.read_csv(fileName, header=None)
    trainingData = fullData.sample(frac=trainingPercent)
    testingData = fullData.drop(trainingData.index)

    trainingData.to_csv(trainingOutput, index=False, header=False)
    testingData.to_csv(testingOutput, index=False, header=False)

    return trainingData, testingData

#Take raw sequences and labels and split it 
def removeLabels(dataFrame: df, labelOutput):
    labels = dataFrame[[dataFrame.columns[1]]].copy()
    labels.to_csv(labelOutput, index=False, header=False)

    data = dataFrame.drop(dataFrame.columns[1], axis=1)
    return data, labels

#Make sequencing files
def createSequencingCSV(dataFrame: df, sequenceOutput):
    with open(sequenceOutput, "w", encoding="utf-8") as f:
        for i, (_, row) in enumerate(dataFrame.iterrows(), start=1):
            row_data = " ".join(str(val) for val in row.values)
            
            f.write(f">seq_{i}\n")
            f.write(f"{row_data}\n")

#Append labels to formatted data
def appendLabelsToData(dataFile, labelFile, outputFile):
    data = pd.read_csv(dataFile)
    labels = pd.read_csv(labelFile)

    labels.columns = ['class']
    data['class'] = labels['class']
    data.to_csv(outputFile, index=False)

if __name__ == "__main__":
    # #Split data
    # trainingData, testingData = splitData("sequences_training.txt", "Training_Set_75%.csv", "Testing_Set_25%.csv", 0.75)

    # #Remove labels
    # unlabeledTrainingData, _ = removeLabels(trainingData, "trainingLabels.csv")
    # unlabeledTestingData, _ = removeLabels(testingData, "testingLabels.csv")

    # #Create sequencing file
    # createSequencingCSV(unlabeledTrainingData, "unlabeledTrainingSequence.fa")
    # createSequencingCSV(unlabeledTestingData, "unlabeledTestingSequence.fa")

    # appendLabelsToData("pfeature_training.csv", "trainingLabels.csv", "trainingRelabeled.csv")
    # appendLabelsToData("pfeature_testing.csv", "testingLabels.csv", "testingRelabeled.csv")

    pass