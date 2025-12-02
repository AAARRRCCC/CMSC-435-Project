# DOCS https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.BorderlineSMOTE.html
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
import pandas as pd
from collections import Counter


def _make_numeric(X: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all feature columns are numeric:
      1. Try to convert string columns to numeric.
      2. For columns that remain non-numeric, factorize them into integer codes.
    """
    X_num = X.copy()

    for col in X_num.columns:
        # If already numeric, skip
        if pd.api.types.is_numeric_dtype(X_num[col]):
            continue

        # Try to convert numeric-like strings
        X_num[col] = pd.to_numeric(X_num[col], errors="ignore")

        # If still not numeric, factorize (label-encode) the categories
        if not pd.api.types.is_numeric_dtype(X_num[col]):
            codes, uniques = pd.factorize(X_num[col], sort=True)
            X_num[col] = codes  # integer codes 0,1,2,...

    return X_num


def main():
    df = pd.read_csv('../Dataset/full_normalized.csv')

    x = df.drop('class', axis=1)
    y = df['class']

    # Make sure all feature columns are numeric before SMOTE
    x_num = _make_numeric(x)

    print(f'Original dataset shape: {Counter(y)}')

    smote = BorderlineSMOTE(
        sampling_strategy={'DNA': 5000, 'RNA': 2000, 'DRNA': 500},
        kind='borderline-1',
        random_state=42
    )
    x_resampled, y_resampled = smote.fit_resample(x_num, y)

    print(f'Resampled dataset shape: {Counter(y_resampled)}')

    df_resampled = pd.DataFrame(x_resampled, columns=x_num.columns)
    df_resampled['class'] = y_resampled

    df_resampled.to_csv('../Dataset/full_normalized_BSMOTE.csv', index=False)


def rm_main(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    x = df.drop('class', axis=1)
    y = df['class']

    # Make sure all feature columns are numeric before SMOTE
    x_num = _make_numeric(x)

    smote = BorderlineSMOTE(
        sampling_strategy={'DNA': 3000, 'RNA': 2000, 'DRNA': 500},
        kind='borderline-1',
        k_neighbors=3,
        random_state=42
    )
    # smote = SMOTETomek(
    #     sampling_strategy={'DNA': 2500, 'RNA': 2000, 'DRNA': 500},
    #     random_state=42
    # )
    x_resampled, y_resampled = smote.fit_resample(x_num, y)

    df_resampled = pd.DataFrame(x_resampled, columns=x_num.columns)
    df_resampled['class'] = y_resampled

    # This write is optional in RapidMiner context; remove if you don't want the side-effect
    df_resampled.to_csv('../Dataset/full_normalized_BSMOTE.csv', index=False)

    return df_resampled


if __name__ == '__main__':
    main()
