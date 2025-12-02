"""
Training-mode normalization wrapper.
Use this in the TRAINING branch of your RapidMiner pipeline.

This script fits a StandardScaler on the training data and saves it
for use in the test phase.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Path where we'll save the scaler
SCALER_PATH = '../TempfilesAndOutput/scaler.pkl'

def rm_main(df):
    """
    Normalize data in TRAINING mode:
    - Fit scaler on this data
    - Save scaler for test phase
    - Transform the data
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

    # FIT the scaler on training data
    print(f"🔧 [TRAIN MODE] Fitting scaler on {len(df)} training samples...")

    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # SAVE the scaler for test phase
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)

    print(f"✓ [TRAIN MODE] Scaler saved to {SCALER_PATH}")
    print(f"✓ [TRAIN MODE] Normalized {len(numerical_cols)} features")
    print(f"✓ [TRAIN MODE] Mean range: [{scaler.mean_.min():.4f}, {scaler.mean_.max():.4f}]")
    print(f"✓ [TRAIN MODE] Std range: [{scaler.scale_.min():.4f}, {scaler.scale_.max():.4f}]")

    return df
