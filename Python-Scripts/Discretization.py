import pandas as pd

# TODO: Implement
def get_splits_to_consider(col, lowerBound, upperBound, labels):
    numsInRange = col[lowerBound : upperBound]
    labelsInRange = labels[lowerBound : upperBound]

    # Sort labels the same way so we can keep matching indices aligned
    numsInRange = numsInRange.sort()

    pass


# Any value in interval splits should include index and higher
def calculate_score(col, intervalSplits):
    sum = 0
    previousIndex = -1
    for i in range(intervalSplits):
        intervalVals = col[previousIndex : intervalSplits[i]] if previousIndex else col[: intervalSplits[i]]
        maxValue = max(intervalVals)
        sum = sum + split_section_score(maxValue, len(intervalVals))
        previousIndex = intervalSplits[i]

    return sum / len(intervalVals)

def split_section_score(max, numContinuous):
    return (max^2) / numContinuous

# TODO: Implement
# Any value in interval splits should include index and higher
def conduct_discretization(df, colIndex, intervalSplits):
    pass

def rm_main(filePath):
    df = pd.read_csv(filePath)
    rows, cols = df.shape
    labels = df.iloc[: cols-1]

    # Discretize each column individually (do not include labels column)
    for i in range(cols - 1):
        intervalSplits = []
        maxScore = 0
        col = df.iloc[:, i]

        # Do this for each split (num lines to split + 1 to find num sections to do this for)
        j = 0
        while j < len(intervalSplits) + 1:

            # Determine bounds and find optimal split index
            lowerBound = 0 if j == 0 else intervalSplits[j]
            upperBound = len(col) - 1 if j == len(intervalSplits) else intervalSplits[j + 1]
            indices = get_splits_to_consider(col, lowerBound, upperBound, labels)

            # Check each index returned and compare to best score
            indexToAdd = -1
            for index in indices:
                score = calculate_score(col, (intervalSplits + [index]).sort())

                # If score is larger, set indexToAdd for adding to intervalSplits and use as flag to continue
                if score > maxScore:
                    maxScore = score
                    indexToAdd = index

            if indexToAdd > -1:
                intervalSplits.append(indexToAdd)
                intervalSplits.sort()
                continue
            
            # Flag was not set, continue to next section and attempt to split
            j = j + 1

        # After determining final splits, discretize data
        df = conduct_discretization(df, i, intervalSplits)





if __name__ == "__main__":
    filePath = "../Dataset/full_normalized_BSMOTE.csv"
    rm_main(filePath)