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

def createSequencingCSV(dataFrame: df, sequenceOutput):
    with open(sequenceOutput, "w", encoding="utf-8") as f:
        for i, (_, row) in enumerate(dataFrame.iterrows(), start=1):
            row_data = " ".join(str(val) for val in row.values)
            
            f.write(f">seq_{i}\n")
            f.write(f"{row_data}\n")

if __name__ == "__main__":
    #Split data
    trainingData, testingData = splitData("sequences_training.txt", "Training_Set_75%.csv", "Testing_Set_25%.csv", 0.75)

    #Remove labels
    unlabeledTrainingData, _ = removeLabels(trainingData, "trainingLabels.csv")
    unlabeledTestingData, _ = removeLabels(testingData, "testingLabels.csv")

    #Create sequencing file
    createSequencingCSV(unlabeledTrainingData, "unlabeledTrainingSequence.txt")
    createSequencingCSV(unlabeledTestingData, "unlabeledTestingSequence.txt")