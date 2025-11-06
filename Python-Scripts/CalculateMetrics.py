import math
import pandas as pd
from pandas import DataFrame as df
from predictions_to_confusion import getConfusionDF

def getSpecificity(TN: int, FP: int) -> float:
    return TN / (TN + FP) if (TN + FP) != 0 else 0.0

def getSensitivity(TP: int, FN: int) -> float:
    return TP / (TP + FN) if (TP + FN) != 0 else 0.0

def getAccuracy(TP: int, TN: int, FP: int, FN: int) -> float:
    return (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0.0

def getMCC(TP: int, TN: int, FP: int, FN: int) -> float:
    bottom = math.sqrt((TP + FP) * (TP + FN) * (TN + FN) * (TN + FP))
    if bottom == 0:
        return 0.0
    top = (TN * TP) - (FN * FP)
    return top / bottom

def main(data: pd.DataFrame):
    rows, cols = data.shape

    if rows != cols:
        print("DataFrame must be square.")
        return []

    results = []

    for i in range(rows):
        TP = data.iloc[i, i]
        FP = data.iloc[:, i].drop(data.index[i]).sum()
        FN = data.iloc[i].drop(data.columns[i]).sum()
        TN = data.to_numpy().sum() - (TP + FP + FN)

        specificity = getSpecificity(TN, FP)
        sensitivity = getSensitivity(TP, FN)
        accuracy = getAccuracy(TP, TN, FP, FN)
        mcc = getMCC(TP, TN, FP, FN)
        label = data.columns[i]

        results.append({
            "Label": label,
            "Specificity": specificity,
            "Sensitivity": sensitivity,
            "Accuracy": accuracy,
            "MCC": mcc
        })

    return results

def rm_main(filepath):
    dataDF = getConfusionDF(filepath)

    results = main(dataDF)

    with open("../TempfilesAndOutput/metrics_output.txt", "w") as f:
        for result in results:
            f.write(
                f"Label: {result['Label']}\n"
                f"\tSpecificity: {result['Specificity']:.3f}\n"
                f"\tSensitivity: {result['Sensitivity']:.3f}\n"
                f"\tAccuracy: {result['Accuracy']:.3f}\n"
                f"\tMCC: {result['MCC']:.3f}\n"
            )

if __name__ == "__main__":
    filepath = "../Dataset/rapidminer_results.csv"
    dataDF = getConfusionDF(filepath)

    results = main(dataDF)
    for result in results:
        print(
            f"Label: {result['Label']}\n"
            f"\tSpecificity: {result['Specificity']:.3f}\n"
            f"\tSensitivity: {result['Sensitivity']:.3f}\n"
            f"\tAccuracy: {result['Accuracy']:.3f}\n"
            f"\tMCC: {result['MCC']:.3f}\n"
        )
