import pandas as pd
import numpy as np

def get_splits_to_consider(col, lowerBound, upperBound, labels):
    # Restrict to range and sort both together
    subset = pd.DataFrame({
        'val': col.iloc[lowerBound:upperBound].reset_index(drop=True),
        'label': labels.iloc[lowerBound:upperBound].reset_index(drop=True)
    }).sort_values(by='val').reset_index(drop=True)

    vals = subset['val']
    labels = subset['label']

    # Candidate splits are midpoints between unique adjacent values with different labels
    candidate_indices = []
    for i in range(1, len(vals)):
        if labels[i] != labels[i - 1]:
            candidate_indices.append(i)

    return candidate_indices

def calculate_caim_score(col, labels, intervalSplits):
    if len(col) == 0:
        return 0.0

    intervalSplits = sorted(intervalSplits)
    num_intervals = len(intervalSplits) + 1
    total_score = 0

    prev = 0
    for split in intervalSplits + [len(col)]:
        # Slice interval
        interval_vals = col.iloc[prev:split]
        interval_labels = labels.iloc[prev:split]

        if len(interval_vals) == 0:
            prev = split
            continue

        # Count per class
        counts = interval_labels.value_counts()
        max_class_count = counts.max()
        total_in_interval = counts.sum()

        # Add term (max^2 / n)
        total_score += (max_class_count ** 2) / total_in_interval
        prev = split

    # Average over number of intervals
    return total_score / num_intervals

def conduct_discretization(df, colIndex, intervalSplits):
    intervalSplits = sorted(intervalSplits)
    col = df.iloc[:, colIndex].sort_values().reset_index(drop=True)

    # Separate data into segments based on intervals
    segments = [-np.inf] + [col.iloc[i] for i in intervalSplits] + [np.inf]
    labels = list(range(len(segments) - 1))
    df.iloc[:, colIndex] = pd.cut(df.iloc[:, colIndex], bins=segments, labels=labels)

    return df

def rm_main(filePath):
    df = pd.read_csv(filePath)
    rows, cols = df.shape
    labels = df.iloc[:, -1]  # Last column is class labels

    for i in range(cols - 1):  # Skip the label column
        col = df.iloc[:, i].sort_values().reset_index(drop=True)
        lbl_sorted = labels.iloc[col.index].reset_index(drop=True)

        intervalSplits = []
        best_score = calculate_caim_score(col, lbl_sorted, intervalSplits)
        improved = True

        while improved:
            improved = False
            candidate_indices = get_splits_to_consider(col, 0, len(col), lbl_sorted)

            best_local_score = best_score
            best_split = None

            for index in candidate_indices:
                test_splits = sorted(intervalSplits + [index])
                score = calculate_caim_score(col, lbl_sorted, test_splits)
                if score > best_local_score:
                    best_local_score = score
                    best_split = index

            # If improvement found, accept new split and continue
            if best_split is not None and best_local_score > best_score:
                intervalSplits.append(best_split)
                intervalSplits.sort()
                best_score = best_local_score
                improved = True

        df = conduct_discretization(df, i, intervalSplits)

    return df


if __name__ == "__main__":
    filePath = "../Dataset/full_normalized_BSMOTE.csv"
    df_out = rm_main(filePath)
    print(df_out.head())
