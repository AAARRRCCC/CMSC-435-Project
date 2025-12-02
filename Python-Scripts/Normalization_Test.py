"""
Test-mode normalization wrapper.
Use this in the TEST branch of your RapidMiner pipeline.

This script loads the scaler fitted on training data and applies it
to the test data (transform only, NO fitting).
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Path where the training scaler was saved
SCALER_PATH = '../TempfilesAndOutput/scaler.pkl'

def rm_main(df):
    """
    Normalize data in TEST mode:
    - Load scaler fitted on training data
    - Transform this data using training statistics
    - NO fitting on test data (prevents leakage)
    - Return normalized data
    """
    df = df.copy()

    # Identify numerical columns (exclude label)
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if 'class' in numerical_cols:
        numerical_cols.remove('class')

    if len(numerical_cols) == 0:
        print("⚠️ Warning: No numerical columns found to normalize")
        return df

    # LOAD the scaler from training phase
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            f"❌ ERROR: Scaler file not found at {SCALER_PATH}\n"
            f"   Make sure the TRAINING normalization ran first!\n"
            f"   Expected file: {os.path.abspath(SCALER_PATH)}"
        )

    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    print(f"🔍 [TEST MODE] Loading scaler from {SCALER_PATH}")

    # ONLY TRANSFORM - do NOT fit!
    df[numerical_cols] = scaler.transform(df[numerical_cols])

    print(f"✓ [TEST MODE] Normalized {len(df)} test samples using TRAINING statistics")
    print(f"✓ [TEST MODE] Applied {len(numerical_cols)} feature transformations")
    print(f"✓ [TEST MODE] Using training mean range: [{scaler.mean_.min():.4f}, {scaler.mean_.max():.4f}]")
    print(f"✓ [TEST MODE] Using training std range: [{scaler.scale_.min():.4f}, {scaler.scale_.max():.4f}]")

    return df
