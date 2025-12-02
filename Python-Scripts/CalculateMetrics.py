import math
import pandas as pd
from pandas import DataFrame as df
from predictions_to_confusion import getConfusionDF
from datetime import datetime

def getSpecificity(TN: int, FP: int) -> float:
    return (100 * TN) / (TN + FP) if (TN + FP) != 0 else 0.0

def getSensitivity(TP: int, FN: int) -> float:
    return (100 * TP) / (TP + FN) if (TP + FN) != 0 else 0.0

def getAccuracy(TP: int, TN: int, FP: int, FN: int) -> float:
    return (100 * (TP + TN)) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0.0

def getMCC(TP: int, TN: int, FP: int, FN: int) -> float:
    bottom = math.sqrt((TP + FP) * (TP + FN) * (TN + FN) * (TN + FP))
    if bottom == 0:
        return 0.0
    top = (TN * TP) - (FN * FP)
    return top / bottom

def getConfusionMatrix(df: pd.DataFrame):
    return df.to_string()

def getOverallAccuracy(data: pd.DataFrame) -> float:
    """
    Calculate overall accuracy from confusion matrix.
    Overall accuracy = (sum of diagonal) / (sum of all elements)
    """
    diagonal_sum = sum(data.iloc[i, i] for i in range(len(data)))
    total_sum = data.to_numpy().sum()
    return (100 * diagonal_sum / total_sum) if total_sum != 0 else 0.0

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
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "Accuracy": accuracy,
            "MCC": mcc
        })

    confusionMatrix = getConfusionMatrix(data)
    overallAccuracy = getOverallAccuracy(data)
    return results, confusionMatrix, overallAccuracy

def rm_main(filepath):
    dataDF = getConfusionDF(filepath)

    results, confusionMatrix, overallAccuracy = main(dataDF)

    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Append mode to preserve history
    with open("../TempfilesAndOutput/metrics_output.txt", "a") as f:
        # Add separator and timestamp header
        f.write("\n" + "="*80 + "\n")
        f.write(f"RUN DATE: {timestamp}\n")
        f.write(f"USING AAC 20 features + DPC ~400 + PAAC, RRI, ACR, PCP, AAI, top 400 features, weighted by information gain, weights normalized before chosen\n")
        f.write(f"NO custom prediction thresholds, \n")
        f.write(f"Stacking with XGBoost[150 rounds, LR=0.05, depth=8], GBTrees, and Random Forest, meta learner = decision tree [maximal depth 30, NO pruning]\n")
        f.write("="*80 + "\n\n")

        # Write overall accuracy first
        f.write(f"OVERALL ACCURACY: {overallAccuracy:.3f}%\n\n")

        # Write per-class metrics
        for result in results:
            f.write(
                f"Label: {result['Label']}\n"
                f"\tSpecificity: {result['Specificity']:.3f}\n"
                f"\tSensitivity: {result['Sensitivity']:.3f}\n"
                f"\tAccuracy: {result['Accuracy']:.3f}\n"
                f"\tMCC: {result['MCC']:.3f}\n"
            )

        # Write confusion matrix
        f.write(f"\n{confusionMatrix}\n")

if __name__ == "__main__":
    filepath = "../Dataset/rapidminer_results.csv"
    dataDF = getConfusionDF(filepath)

    results, confusionMatrix, overallAccuracy = main(dataDF)

    print(f"OVERALL ACCURACY: {overallAccuracy:.3f}%\n")

    for result in results:
        print(
            f"Label: {result['Label']}\n"
            f"\tSpecificity: {result['Specificity']:.3f}\n"
            f"\tSensitivity: {result['Sensitivity']:.3f}\n"
            f"\tAccuracy: {result['Accuracy']:.3f}\n"
            f"\tMCC: {result['MCC']:.3f}\n"
        )
