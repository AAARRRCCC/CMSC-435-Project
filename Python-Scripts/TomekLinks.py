import pandas as pd

def distance(row1, row2):
    # Not sure how to do distance, just summing all the cols rn (not label)
    return sum(abs(row1[:-1] - row2[:-1]))

# Defaults to only remove majority class if link is found
def findTomekLinks(df, removeBoth=False):
    indicesToRemove = set()
    numRows = len(df)
    
    # For each row
    for i in range(numRows):
        rowi = df.iloc[i]
        minDistance = float('inf')
        nearestIndex = None
        
        # For each other item to find NN
        for j in range(numRows):
            if i == j:
                continue
            rowJ = df.iloc[j]
            distIJ = distance(rowi.values, rowJ.values)
            if distIJ < minDistance:
                minDistance = distIJ
                nearestIndex = j
        
        # Check if nearest neighbor is mutual
        if nearestIndex is not None:
            rowNN = df.iloc[nearestIndex]
            
            # Find nearest neighbor of rowNN
            minDistNN = float('inf')
            nearestNNindex = None
            for k in range(numRows):
                if nearestIndex == k:
                    continue
                rowK = df.iloc[k]
                distNN = distance(rowNN.values, rowK.values)
                if distNN < minDistNN:
                    minDistNN = distNN
                    nearestNNindex = k
            
            # If mutual nearest neighbors are different classes then remove
            if nearestNNindex == i and rowi.iloc[-1] != rowNN.iloc[-1]:

                # Determine which to remove
                if removeBoth is True:
                    indicesToRemove.add(i) # Remove OG
                    indicesToRemove.add(nearestIndex) # Remove NN to OG
                else:
                    # This only removes majority class
                    classes = [rowi.iloc[-1], rowNN.iloc[-1]]
                    majorityClass = max(classes, key=classes.count)
                    if rowi.iloc[-1] == majorityClass:
                        indicesToRemove.add(i) # Remove OG
                    else:
                        indicesToRemove.add(nearestIndex) # Remove NN to OG
    
    return indicesToRemove

def removeTomekLinks(inputFile, outputFile):
    df = pd.read_csv(inputFile)
    
    toRemove = findTomekLinks(df)
    print(f"Removed {len(toRemove)} rows")
    
    dfUnlinked = df.drop(index=toRemove)
    dfUnlinked.to_csv(outputFile, index=False)

if __name__ == "__main__":
    inputFile = "somePath/IG"
    outputFile = "somePath/IG"
    removeTomekLinks(inputFile, outputFile, False) # Last arg removes both items in link if true, or only majority class in link if false