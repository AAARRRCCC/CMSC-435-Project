#!/usr/bin/env python3
"""
Optimized CAIM discretization

Usage:
    python caim_optimized.py path/to/your.csv

Notes:
- Expects class labels in the last column.
- Numeric columns will be discretized; non-numeric columns are left unchanged.
- Produces printed progress and returns a discretized DataFrame (also prints head).
"""

from typing import List, Tuple
import time
import sys

import numpy as np
import pandas as pd


def build_cumulative_counts(labels: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """
    Build cumulative counts array of shape (num_classes, n)
    cum_counts[j, i] = number of occurrences of classes[j] in labels[:i+1]
    """
    n = labels.shape[0]
    num_classes = classes.shape[0]
    cum_counts = np.empty((num_classes, n), dtype=np.int32)
    for j, c in enumerate(classes):
        mask = (labels == c).astype(np.int32)
        cum_counts[j] = np.cumsum(mask)
    return cum_counts


def interval_class_counts(cum_counts: np.ndarray, start: int, end: int) -> np.ndarray:
    """
    Return class counts for interval [start, end) using cumulative arrays.
    start inclusive, end exclusive (end > start)
    """
    # counts over [start, end) = cum[end-1] - (cum[start-1] if start>0 else 0)
    if start == 0:
        return cum_counts[:, end - 1].copy()
    else:
        return (cum_counts[:, end - 1] - cum_counts[:, start - 1]).copy()


def caim_term_from_counts(counts: np.ndarray) -> float:
    """
    CAIM term for a single interval: (max_class_count^2) / total_in_interval
    """
    total = counts.sum()
    if total == 0:
        return 0.0
    maxc = counts.max()
    return (maxc * maxc) / float(total)


def find_candidates_in_interval(values: np.ndarray, labels: np.ndarray, start: int, end: int) -> np.ndarray:
    """
    Find candidate split positions within [start, end).
    Candidate positions are indices p where start < = p < end and:
       - values[p] != values[p-1]  (distinct adjacent values)
       - labels[p] != labels[p-1]  (label change across adjacency)
    Return absolute split indices p (where left interval = [start, p), right = [p, end))
    """
    if end - start <= 1:
        return np.array([], dtype=np.int32)
    # check positions from start+1 .. end-1 inclusive
    left = values[start + 1: end]
    right = values[start: end - 1]
    vals_differ = left != right
    lab_left = labels[start + 1: end]
    lab_right = labels[start: end - 1]
    labs_differ = lab_left != lab_right
    mask = vals_differ & labs_differ
    # positions relative to start: indices where mask True => rel_idx + 1 relative to start
    rel_idxs = np.nonzero(mask)[0] + 1
    if rel_idxs.size == 0:
        return np.array([], dtype=np.int32)
    abs_idxs = (start + rel_idxs).astype(np.int32)
    return abs_idxs


def discretize_column_by_splits(original_col: pd.Series,
                                values_sorted: np.ndarray,
                                split_positions: List[int]) -> pd.Series:
    """
    Convert split_positions (indices into sorted values) to numeric thresholds,
    then replace each value by the numeric center of its bin.

    Result: column remains purely numeric (float), so later normalization
    can still work on these discretized features.
    """
    # If no splits were found, just return the original numeric column
    if len(split_positions) == 0:
        return original_col.astype(float)

    # 1) Compute thresholds as midpoints between sorted adjacent values
    thresholds: List[float] = []
    for p in sorted(split_positions):
        v_left = values_sorted[p - 1]
        v_right = values_sorted[p]
        thr = (float(v_left) + float(v_right)) / 2.0
        thresholds.append(thr)

    # 2) We will use these thresholds as internal bin edges.
    #    To get finite edges for the first and last bins, we look at the
    #    actual min/max of the original column.
    col_values = original_col.to_numpy(dtype=float)
    col_min = np.nanmin(col_values)
    col_max = np.nanmax(col_values)

    # Edges for center computation: [left_edge_0, edge_1, ..., edge_k, right_edge_last]
    # with internal edges = thresholds
    edges = [col_min] + thresholds + [col_max]

    # 3) Compute central value for each bin
    #    Bin i has range [edges[i], edges[i+1]] -> center = (left + right)/2
    centers = []
    for left, right in zip(edges[:-1], edges[1:]):
        centers.append((left + right) / 2.0)
    centers = np.asarray(centers, dtype=float)

    # 4) Assign each original value to a bin by thresholds
    #    np.digitize returns bin index in [0, len(thresholds)] for each value.
    #    thresholds are the internal edges between bins.
    bin_indices = np.digitize(col_values, thresholds, right=False)

    # Map bin index -> central value
    new_values = centers[bin_indices]

    # Return as a float Series with original index preserved
    return pd.Series(new_values, index=original_col.index, dtype=float)



def caim_discretize_for_column(values: np.ndarray, labels: np.ndarray,
                               min_gain: float = 1e-6, max_intervals: int = None) -> List[int]:
    """
    Main CAIM algorithm for a single numeric column.
    Returns split positions as indices into the sorted array (integers p where left = [..p), right = [p..))
    """
    n = values.shape[0]
    if n == 0:
        return []

    # unique label classes and cumulative counts
    classes = np.unique(labels)
    num_classes = classes.shape[0]

    if max_intervals is None:
        max_intervals = num_classes  # CAIM common heuristic

    cum_counts = build_cumulative_counts(labels, classes)

    # Keep intervals as list of (start, end) with start inclusive, end exclusive
    intervals: List[Tuple[int, int]] = [(0, n)]
    # total_score is sum of (max^2 / nj) across intervals
    total_score = 0.0
    # compute initial total_score
    initial_counts = interval_class_counts(cum_counts, 0, n)
    total_score = caim_term_from_counts(initial_counts)
    current_k = 1

    split_positions: List[int] = []

    # main loop: try to add one split per iteration (greedy). Stop when no positive gain or max_intervals reached.
    while True:
        best_gain = 0.0
        best_interval_idx = None
        best_split_pos = None
        best_new_terms = None  # store (term_before, term_left, term_right) for quick update

        current_avg = total_score / float(current_k)

        # Consider each current interval and find best split inside it
        for iv_idx, (start, end) in enumerate(intervals):
            if end - start <= 1:
                continue

            # candidates inside this interval
            candidates = find_candidates_in_interval(values, labels, start, end)
            if candidates.size == 0:
                continue

            # precompute counts for the whole interval
            counts_interval = interval_class_counts(cum_counts, start, end)
            term_before = caim_term_from_counts(counts_interval)

            # Evaluate every candidate in this interval (we will typically have few)
            # We'll compute left/right counts using cumulative arrays
            for p in candidates:
                left_counts = interval_class_counts(cum_counts, start, p)
                right_counts = interval_class_counts(cum_counts, p, end)
                term_left = caim_term_from_counts(left_counts)
                term_right = caim_term_from_counts(right_counts)
                new_total_score = total_score - term_before + (term_left + term_right)
                new_k = current_k + 1
                new_avg = new_total_score / float(new_k)
                gain = new_avg - current_avg

                # Accept if best gain so far
                if gain > best_gain:
                    best_gain = gain
                    best_interval_idx = iv_idx
                    best_split_pos = int(p)
                    best_new_terms = (term_before, term_left, term_right)

        # If no gain above threshold or we reached max_intervals, stop
        if best_gain <= min_gain or current_k >= max_intervals or best_split_pos is None:
            break

        # Apply chosen split
        # Split the interval into two: intervals[best_interval_idx] -> (start, best_split_pos), (best_split_pos, end)
        start, end = intervals[best_interval_idx]
        left = (start, best_split_pos)
        right = (best_split_pos, end)
        # Replace the interval in the list with two new ones (keep order)
        intervals[best_interval_idx:best_interval_idx + 1] = [left, right]
        # Record split
        split_positions.append(best_split_pos)
        # Update total_score and counters
        term_before, term_left, term_right = best_new_terms
        total_score = total_score - term_before + (term_left + term_right)
        current_k += 1

        # optional early stop: if we reached number of unique values or other limits
        if current_k >= max_intervals:
            break

    # Return sorted split positions
    split_positions.sort()
    return split_positions


def rm_main(df: pd.DataFrame,
                      min_gain: float = 1e-6,
                      verbose: bool = True) -> pd.DataFrame:
    """
    Run optimized CAIM on CSV at file_path.
    Discretizes numeric columns using CAIM and returns the modified DataFrame.
    """
    # df = pd.read_csv(file_path)
    rows, cols = df.shape

    # assume last column is label
    label_col = df.columns[-1]
    labels_series = df[label_col]
    # Convert labels to numpy array (we will reindex per sorted order per column)
    # For consistency, we'll treat labels as they correspond row-wise to values (no global sorting)
    # For each numeric column, we will sort the column values and reorder labels accordingly before running CAIM.

    df_out = df.copy()

    numeric_cols = df_out.select_dtypes(include=[np.number]).columns.tolist()
    # exclude label column if numeric
    if label_col in numeric_cols:
        numeric_cols.remove(label_col)

    # iterate numeric columns
    for idx, col_name in enumerate(numeric_cols):
        col_series = df_out[col_name]
        # Sort column with stable index mapping so we can reorder labels
        order = col_series.argsort(kind='mergesort')  # stable sort preserves order of equal values
        vals_sorted = col_series.to_numpy()[order]
        labs_sorted = labels_series.to_numpy()[order]

        # Run CAIM for this column
        splits = caim_discretize_for_column(vals_sorted, labs_sorted, min_gain=min_gain)

        # Discretize original column using thresholds from splits
        new_col = discretize_column_by_splits(df_out[col_name], vals_sorted, splits)
        df_out[col_name] = new_col

    return df_out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python caim_optimized.py path/to/file.csv")
        sys.exit(1)

    path = sys.argv[1]
    out_df = rm_main(path, min_gain=1e-6, verbose=True)
    print(out_df.head())
