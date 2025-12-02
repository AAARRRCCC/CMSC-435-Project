# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a machine learning pipeline for protein sequence classification (DNA, RNA, DRNA, nonDRNA). The project integrates RapidMiner workflows (.rmp files) with Python-based data preprocessing, feature extraction using pfeature, and model evaluation.

## Core Architecture

### Data Flow Pipeline

The pipeline follows this sequence:

1. **Data Preparation** (`SplitData.py`): Splits raw sequences and labels from `StartingData.txt`
   - `removeLabels()`: Separates sequences from class labels
   - `createSequencingCSV()`: Converts data to FASTA format for pfeature processing
   - Entry point: `rm_main()` takes `dataFilePath` and outputs paths to sequenced file and labels

2. **Feature Extraction** (`pfeature_standalone/runPFeature.py`): Generates feature vectors from sequences
   - Runs `pfeature_comp.py` with DPC (Dipeptide Composition) method
   - Returns DataFrame of numerical features
   - Entry point: `rm_main(pfeatureinputfilepath)`

3. **Label Restoration** (`GlueLabels.py`): Re-attaches class labels to feature data
   - `appendLabelsToData()`: Merges labels back with processed features
   - Entry point: `rm_main(data_df, labelsFilePath)`

4. **Normalization** (`Normalization.py`): Z-score normalization of numerical features
   - Uses `StandardScaler` from scikit-learn
   - Entry point: `rm_main(df)` takes DataFrame and returns normalized DataFrame

5. **Class Balancing** (`Oversample.py`): Handles class imbalance using BorderlineSMOTE
   - Resamples to target counts: DNA=2000, RNA=2000, DRNA=500
   - Entry point: `rm_main(df)` returns resampled DataFrame

6. **RapidMiner Integration**: Pipeline files (`.rmp`) orchestrate the workflow
   - Most recent: `Pipeline11-30.rmp`
   - These coordinate Python scripts and model training

7. **Evaluation** (`CalculateMetrics.py`): Computes classification metrics
   - Processes RapidMiner prediction outputs
   - Calculates sensitivity, specificity, accuracy, MCC per class
   - Entry point: `rm_main(filepath)` writes metrics to `TempfilesAndOutput/metrics_output.txt`

### Key Design Patterns

**Dual Entry Points**: Most Python scripts have two execution modes:
- `main()`: Standalone execution with hardcoded paths
- `rm_main()`: RapidMiner integration mode accepting DataFrame arguments

**Path Convention**: Scripts expect data in `../Dataset/` relative to `Python-Scripts/` directory structure:
- `Dataset/StartingData.txt`: Original raw data
- `Dataset/Unfinished/`: Intermediate processing outputs
- `TempfilesAndOutput/`: Final results and metrics

## Development Commands

### Setup
```bash
pip install -r requirements.txt
```

### Running the Pipeline

The pipeline is primarily orchestrated through RapidMiner (.rmp files), but individual components can be tested:

**Feature extraction (standalone)**:
```bash
cd pfeature_standalone
python runPFeature.py ../Dataset/Unfinished/pfeatureSequenced.fa
```

**Normalization (standalone)**:
```bash
cd Python-Scripts
python Normalization.py <input_csv_path>
```

**Calculate metrics from predictions**:
```bash
cd Python-Scripts
python CalculateMetrics.py
# Reads from: ../Dataset/rapidminer_results.csv
# Outputs to: ../TempfilesAndOutput/metrics_output.txt
```

**Discretization (CAIM algorithm)**:
```bash
cd Python-Scripts
python Discretization.py path/to/file.csv
```

## Additional Utilities

**Undersampling**: `Undersample.py` - Reduces specific class representation by percentage
**Tomek Links**: `TomekLinks.py` - Removes boundary examples between classes (naive implementation)
**Confusion Matrix**: `predictions_to_confusion.py` - Converts RapidMiner predictions to confusion matrix DataFrame

## Important Implementation Notes

- All normalization uses Z-score (StandardScaler) for better ML performance
- BorderlineSMOTE is configured with fixed random_state=42 for reproducibility
- Discretization uses optimized CAIM algorithm with cumulative count arrays
- The confusion matrix expects labels in order: ['nonDRNA','RNA', 'DNA', 'DRNA']
- When modifying data processing scripts, ensure both `main()` and `rm_main()` entry points are updated
- File paths in standalone mode are relative to `Python-Scripts/` directory
- pfeature outputs are temporary (CSVTEMP.csv) and should be processed immediately

## Git Workflow

- Main branch: `main`
- Current active branches include feature branches (e.g., `Ryan2`, `Eli`)
- Pipeline iterations are tracked by date suffix (Pipeline11-18.rmp, Pipeline11-20.rmp, Pipeline11-30.rmp)
