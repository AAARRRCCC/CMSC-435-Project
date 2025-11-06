# DOCS https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.BorderlineSMOTE.html
from imblearn.over_sampling import BorderlineSMOTE
import pandas as pd
from collections import Counter



df = pd.read_csv('../Dataset/full_normalized.csv')
x = df.drop('class', axis=1)
y = df['class']

print(f'Original dataset shape: {Counter(y)}')

smote = BorderlineSMOTE(sampling_strategy={'DNA':2000, 'RNA':2000, 'DRNA':500}, kind='borderline-1', random_state=42) 
x_resampled, y_resampled = smote.fit_resample(x, y)


print(f'Resampled dataset shape: {Counter(y_resampled)}')

df_resampled = pd.DataFrame(x_resampled, columns=x.columns)
df_resampled['class'] = y_resampled

df_resampled.to_csv('../Dataset/full_normalized_BSMOTE.csv', index=False)


def rm_main(filepath):

    df = pd.read_csv(filepath)

    smote = BorderlineSMOTE(sampling_strategy={'DNA':2000, 'RNA':2000, 'DRNA':500}, kind='borderline-1', random_state=42) 
    x_resampled, y_resampled = smote.fit_resample(x, y)

    df_resampled = pd.DataFrame(x_resampled, columns=x.columns)
    df_resampled['class'] = y_resampled

    df_resampled.to_csv('../Dataset/full_normalized_BSMOTE.csv', index=False)

    return df_resampled