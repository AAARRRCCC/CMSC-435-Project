import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def getConfusionDF(filepath):

    df = pd.read_csv(filepath)
    y_true = df['class']
    y_pred = df['prediction(class)']

    cm = confusion_matrix(y_pred, y_true, labels=['nonDRNA','RNA', 'DNA', 'DRNA'])
    cm_df = pd.DataFrame(cm, index=['nonDRNA','RNA', 'DNA', 'DRNA'], columns=['nonDRNA','RNA', 'DNA', 'DRNA'])
    #print(cm_df)
    return cm_df