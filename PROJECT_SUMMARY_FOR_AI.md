# Protein Sequence Classification Project - Complete Data Reference

## Project Overview

A machine learning pipeline for classifying protein sequences into four categories:
- **nonDRNA** - Non-DNA-binding proteins (most common, ~89% of dataset)
- **RNA** - RNA-binding proteins (~6% of dataset)
- **DNA** - DNA-binding proteins (~4% of dataset)
- **DRNA** - Dual RNA/DNA-binding proteins (rarest, <1% of dataset)

The pipeline integrates RapidMiner workflows with Python preprocessing, using pfeature for feature extraction and various ML algorithms for classification.

---

## Data Pipeline Architecture

```
Raw Data (StartingData.txt)
    ↓
[SplitData.py] - Separates sequences from labels
    ↓
Unlabeled Sequences (FASTA format)
    ↓
[pfeature] - Extracts DPC (Dipeptide Composition) features
    ↓
Feature Vectors (400 DPC features)
    ↓
[GlueLabels.py] - Re-attaches class labels
    ↓
Labeled Feature Data
    ↓
[Normalization.py] - Z-score normalization (StandardScaler)
    ↓
Normalized Data
    ↓
[Oversample.py] - BorderlineSMOTE class balancing
    ↓
Balanced Dataset (DNA=2000, RNA=2000, DRNA=500)
    ↓
[RapidMiner] - Model training & prediction
    ↓
Predictions CSV
    ↓
[CalculateMetrics.py] - Performance evaluation
    ↓
Final Metrics Report
```

---

## Data Formats & Samples

### 1. Raw Input Data (StartingData.txt)

**Format:** CSV with sequence and label
**Columns:** `sequence,class`

**Sample rows:**
```
MLKQVEIFTDGSCLGNPGPGGYGAILRYRGREKTFSAGYTRTTNNRMELMAAIVALEALKEHCEVILSTDSQYVRQGITQWIHNWKKRGWKTADKKPVKNVDLWQRLDAALGQHQIKWEWVKGHAGHPENERCDELARAAAMNPTLEDTGYQVEV,nonDRNA

MEQKKMKYLENLVGKTPMLELIFDYKGEERRIFVKNESYNLTGSIKDRMAFYTLKKAYEKNEIKKGAPIVEATSGNTGIAFSAMGAILGHPVIIYMPDWMSEERKSLIRSFGAKIILVSRKEGGFLGSIEKTKEFAKNNPDTYLPSQFSNLYNSEAHYYGIGLEIVNEMKSLNLNIDGFVAGVGTGGTVMGIGKRIKENFSNAKICPLEPLNSPTLSTGYKVAKHRIEGISDEFIPDLVKLDKLDNVVSVDDGDAIVMAQKLAKCGLGVGISSGANFIGALMLQNKLGKDSVIVTVFPDDNKKYLSTDLMREEKVKEDFLSKDITLKEIKNVLRVI,nonDRNA

MAERGGDGGEGERFNPGELRMAQQQALRFRGPAPPPNAVMRGPPPLMRPPPPFGMMRGPPPPPRPPFGRPPFDPNMPPMPPPGGIPPPMGPPHLQRPPFMPPPMGAMPPPPGMMFPPGMPPGTAPGAPALPPTEEIWVENKTPDGKVYYYNARTRESAWTKPDGVKVIQQSELTPMLAAQAQVQAQAQAQAQAQAQAQAQAQAQAQAQAQAQAQAQAQAQAQAQAQAQAQAQAQAQAQAQAQAQAQAQAQAQVQAQAVGAPTPTTSSPAPAVSTSTPTSTPSSTTATTTTATSVAQTVSTPTTQDQTPSSAVSVATPTVSVSAPAPTATPVQTVPQPHPQTLPPAVPHSVPQPAAAIPAFPPVMVPPFRVPLPGMPIPLPGVAMMQIVSCPYVKTVATTKTGVLPGMAPPIVPMIHPQVAIAASPATLAGATAVSEWTEYKTADGKTYYYNNRTLESTWEKPQELKEKEKLDEKIKEPIKEASEEPLPMETEEEDPKEEPVKEIKEEPKEEEMTEEEKAAQKAKPVATTPIPGTPWCVVWTGDERVFFYNPTTRLSMWDRPDDLIGRADVDKIIQEPPHKKGLEDMKKLRHPAPTMLSIQKWQFSMSAIKEEQELMEEMNEDEPIKAKKRKRDDNKDIDSEKEAAMEAEIKAARERAIVPLEARMKQFKDMLLERGVSAFSTWEKELHKIVFDPRYLLLNPKERKQVFDQYVKTRAEEERREKKNKIMQAKEDFKKMMEEAKFNPRATFSEFAAKHAKDSRFKAIEKMKDREALFNEFVAAARKKEKEDSKTRGEKIKSDFFELLSNHHLDSQSRWSKVKDKVESDPRYKAVDSSSMREDLFKQYIEKIAKNLDSEKEKELERQARIEASLREREREVQKARSEQTKEIDREREQHKREEAIQNFKALLSDMVRSSDVSWSDTRRTLRKDHRWESGSLLEREEKEKLFNEHIEALTKKKREHFRQLLDETSAITLTSTWKEVKKIIKEDPRCIKFSSSDRKKQREFEEYIRDKYITAKADFRTLLKETKFITYRSKKLIQESDQHLKDVEKILQNDKRYLVLDCVPEERRKLIVAYVDDLDRRGPPPPPTASEPTRRSTK,nonDRNA

MARDATKLEATVAKLKKHWAESAPRDMRAAFSADPGRFGRYSLCLDDLLFDWSKCRVNDETMALLKELAVAADVEGRRAAMFAGEHINNTEDRAVLHVALRDTSSKEVLVDGHNVLPDVKHVLDRMAAFADGIRSGALKGATGRKITDIVNIGIGGSDLGPVMATLALAPYHDEPRAHFVSNIDGAHIADTLSPLDPASTLIIVASKTFTTIETMTNAQTARKWVADTLGEAAVGAHFAAVSTALDKVAAFGIPEDRVFGFWDWVGGRYSVWSAIGLPVMIAVGPDNFRKFLAGAHAMDVHFRDAPLEKNLPVMLGLIGYWHRAICGYGSRAIIPYDQRLSRLPAYLQQLDMESNGKSVTLDGKPVSGPTGPVVWGEPGTNGQHAFFQLLHQGTDTIPLEFIVAAKGHEPTLDHQHEMLMANCLAQSEALMKGRTLDEARAQLQAKNLPASQVERIAPHRVFSGNRPSLTLIHDMLDPYTLGRLIALYEHRVFVEAQIFGINAFDQWGVELGKELATELLPVVSGKEGASGRDASTQGLVAHLHARRKA,nonDRNA
```

**Key characteristics:**
- Amino acid sequences of varying lengths (20-2000+ characters)
- Standard 20 amino acid alphabet (ACDEFGHIKLMNPQRSTVWY)
- Highly imbalanced classes
- Some sequences extremely long (>1500 amino acids)

---

### 2. Intermediate: pfeature FASTA Format

**Format:** FASTA (sequence identifier + sequence on separate lines)

**Sample:**
```
>seq_1
MLKQVEIFTDGSCLGNPGPGGYGAILRYRGREKTFSAGYTRTTNNRMELMAAIVALEA...
>seq_2
MEQKKMKYLENLVGKTPMLELIFDYKGEERRIFVKNESYNLTGSIKDRMAFYTLKKAY...
>seq_3
MTILFQLALAALVILSFVMVIGVPVAYASPQDWDRSKQLIFLGSGLWIALVLVVGVLN...
```

**Purpose:** Input format required by pfeature tool

---

### 3. Feature Vectors (labeledData.csv)

**Format:** CSV with 400 DPC features + class label
**Columns:** `DPC1_AA`, `DPC1_AC`, ..., `DPC1_YY`, `class` (401 total columns)

**Sample rows (first 20 columns shown):**
```
DPC1_AA,DPC1_AC,DPC1_AD,DPC1_AE,DPC1_AF,DPC1_AG,DPC1_AH,DPC1_AI,DPC1_AK,DPC1_AL,DPC1_AM,DPC1_AN,DPC1_AP,DPC1_AQ,DPC1_AR,DPC1_AS,DPC1_AT,DPC1_AV,DPC1_AW,DPC1_AY,...,class
0.0,0.0,0.0,0.0,0.6,0.3,0.3,0.6,1.49,0.3,0.3,0.3,0.3,0.3,0.0,0.0,0.3,0.0,0.0,0.3,...,nonDRNA
1.64,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,4.92,0.0,0.0,0.0,0.0,0.0,1.64,0.0,0.0,0.0,1.64,...,nonDRNA
0.83,0.0,0.55,0.55,0.28,1.11,0.28,0.55,0.83,0.28,0.28,0.28,0.28,0.55,0.55,0.28,0.55,0.55,0.0,0.0,...,nonDRNA
```

**Feature explanation:**
- **DPC (Dipeptide Composition):** Frequency of each possible amino acid pair
- 20 amino acids × 20 amino acids = 400 possible dipeptides
- Each feature represents percentage of that dipeptide in the sequence
- Example: `DPC1_AA` = percentage of "AA" dipeptides, `DPC1_AG` = percentage of "AG" dipeptides

**Value ranges:**
- Raw: 0.0 to ~10.0 (percentage values)
- After normalization: approximately -3.0 to +10.0 (z-scores)

---

### 4. Normalized Data (full_normalized_BSMOTE.csv)

**Format:** Same 401 columns, values are z-score normalized

**Sample (normalized values shown):**
```
DPC1_AA,DPC1_AC,...,class
-0.9909324795108415,-0.46385599291387736,...,nonDRNA
0.8040529470616689,-0.46385599291387736,...,nonDRNA
-0.08249473313572954,-0.46385599291387736,...,nonDRNA
```

**Normalization details:**
- Method: StandardScaler (z-score normalization)
- Formula: `z = (x - mean) / std_dev`
- Applied per-feature across all samples
- Mean ≈ 0, Standard deviation ≈ 1 for each feature

**After oversampling:**
- Original dataset: ~7800 nonDRNA, ~500 RNA, ~400 DNA, ~40 DRNA
- Oversampled targets: ~7800 nonDRNA, 2000 RNA, 2000 DNA, 500 DRNA
- Total rows after BSMOTE: ~12,300 samples

---

### 5. Predictions Output (predictions_output.csv)

**Format:** Full feature data + confidence scores + prediction

**Columns:**
- All 400 DPC features (normalized)
- `class` - true label
- `confidence(nonDRNA)` - model confidence for nonDRNA class
- `confidence(RNA)` - model confidence for RNA class
- `confidence(DNA)` - model confidence for DNA class
- `confidence(DRNA)` - model confidence for DRNA class
- `prediction(class)` - predicted label

**Sample row (abbreviated):**
```
DPC1_AA,...,class,confidence(nonDRNA),confidence(RNA),confidence(DNA),confidence(DRNA),prediction(class)
-0.065,...,nonDRNA,0.5482,0.1554,0.1523,0.1441,nonDRNA
0.0238,...,nonDRNA,0.5336,0.1587,0.1605,0.1472,nonDRNA
```

**Confidence interpretation:**
- Values sum to 1.0 (100%)
- Higher confidence = stronger prediction
- Typical correct predictions: 0.50-0.70 for predicted class
- Misclassifications often show: 0.30-0.45 (low confidence)

---

### 6. Metrics Output (metrics_output.txt)

**Format:** Text report with per-class metrics

**Sample output:**
```
================================================================================
RUN DATE: 2025-11-30 18:10:52
USING RANDOM FOREST, no discretization, maximal depth 200
================================================================================

OVERALL ACCURACY: 89.106%

Label: nonDRNA
	Specificity: 50.825
	Sensitivity: 90.790
	Accuracy: 89.413
	MCC: 0.246
Label: RNA
	Specificity: 95.101
	Sensitivity: 43.154
	Accuracy: 93.678
	MCC: 0.264
Label: DNA
	Specificity: 95.764
	Sensitivity: 35.593
	Accuracy: 95.360
	MCC: 0.124
Label: DRNA
	Specificity: 99.772
	Sensitivity: 66.667
	Accuracy: 99.761
	MCC: 0.246

         nonDRNA  RNA  DNA  DRNA
nonDRNA     7709  413  351    18
RNA          118  104   19     0
DNA           30    6   21     2
DRNA           1    0    0     2
```

**Metric definitions:**
- **Specificity:** True Negative Rate - % of non-members correctly rejected
- **Sensitivity (Recall):** True Positive Rate - % of members correctly identified
- **Accuracy:** Overall correctness for that class
- **MCC (Matthews Correlation Coefficient):** Balanced measure (-1 to +1, 0 = random)
- **Confusion Matrix:** Rows = actual, Columns = predicted

---

## Key Python Scripts

### 1. SplitData.py
**Location:** `Python-Scripts/SplitData.py`

**Functions:**
- `removeLabels(filePath, labelOutput)` - Separates sequences from class labels
- `createSequencingCSV(dataFrame, sequenceOutput)` - Converts to FASTA format
- `rm_main(dataFilePath)` - RapidMiner entry point

**Input:** StartingData.txt (CSV with sequences and labels)
**Output:**
- `pfeatureSequenced.fa` (FASTA format sequences)
- `trainingLabels.csv` (separated labels)

---

### 2. runPFeature.py
**Location:** `pfeature_standalone/runPFeature.py`

**Function:**
- `rm_main(pfeatureinputfilepath)` - Runs pfeature_comp.py for DPC extraction

**Input:** FASTA file with protein sequences
**Output:** DataFrame with 400 DPC features (no labels)
**Method:** Dipeptide Composition (DPC)

**Key command:**
```python
subprocess.run("python pfeature_comp.py -i ../Dataset/Unfinished/pfeatureSequenced.fa -o CSVTEMP.csv -j dpc")
```

---

### 3. GlueLabels.py
**Location:** `Python-Scripts/GlueLabels.py`

**Function:**
- `appendLabelsToData(dataDF, labelsFilePath)` - Merges labels back with features

**Input:**
- Feature DataFrame (from pfeature)
- Labels CSV file
**Output:** Combined DataFrame with features + class column

---

### 4. Normalization.py
**Location:** `Python-Scripts/Normalization.py`

**Function:**
- `rm_main(df)` - Z-score normalizes all numerical features

**Method:** StandardScaler from scikit-learn
**Formula:** `(value - mean) / std_deviation`
**Applied to:** All 400 DPC feature columns
**Preserves:** Class label column unchanged

---

### 5. Oversample.py
**Location:** `Python-Scripts/Oversample.py`

**Function:**
- `rm_main(df)` - Resamples minority classes using BorderlineSMOTE

**Target class counts:**
- DNA: 2000 samples
- RNA: 2000 samples
- DRNA: 500 samples
- nonDRNA: unchanged (~7800)

**Algorithm:** BorderlineSMOTE (variant of SMOTE)
**Random state:** 42 (for reproducibility)

---

### 6. CalculateMetrics.py
**Location:** `Python-Scripts/CalculateMetrics.py`

**Functions:**
- `getSpecificity(TN, FP)` - Calculate specificity
- `getSensitivity(TP, FN)` - Calculate sensitivity/recall
- `getAccuracy(TP, TN, FP, FN)` - Calculate accuracy
- `getMCC(TP, TN, FP, FN)` - Calculate Matthews Correlation Coefficient
- `rm_main(filepath)` - Process predictions and output metrics

**Input:** predictions_output.csv (from RapidMiner)
**Output:** metrics_output.txt (formatted report + confusion matrix)

---

## RapidMiner Pipeline Files

**Latest version:** `Pipeline11-30.rmp`
**Location:** Project root directory

**Pipeline stages:**
1. Execute SplitData.py
2. Execute runPFeature.py
3. Execute GlueLabels.py
4. Execute Normalization.py
5. Execute Oversample.py
6. Train ML model (Random Forest, Gradient Boosted Trees, etc.)
7. Make predictions
8. Execute CalculateMetrics.py

---

## Class Distribution Challenges

### Original dataset (highly imbalanced):
- **nonDRNA:** ~7,800 samples (89%)
- **RNA:** ~500 samples (6%)
- **DNA:** ~400 samples (4%)
- **DRNA:** ~40 samples (<1%)

### After BorderlineSMOTE:
- **nonDRNA:** ~7,800 samples (unchanged)
- **RNA:** 2,000 samples (oversampled)
- **DNA:** 2,000 samples (oversampled)
- **DRNA:** 500 samples (oversampled)

**Impact on metrics:**
- High specificity for minority classes (>95%)
- Low sensitivity for minority classes (20-70%)
- Overall accuracy dominated by nonDRNA performance (~89%)
- MCC provides more balanced view (best: 0.264 for RNA)

---

## Current Performance (Best Model)

**Algorithm:** Random Forest (depth=200, no discretization)
**Overall Accuracy:** 89.1%

**Per-class performance:**
| Class    | Sensitivity | Specificity | MCC   |
|----------|-------------|-------------|-------|
| nonDRNA  | 90.8%       | 50.8%       | 0.246 |
| RNA      | 43.2%       | 95.1%       | 0.264 |
| DNA      | 35.6%       | 95.8%       | 0.124 |
| DRNA     | 66.7%       | 99.8%       | 0.246 |

**Key challenges:**
- Minority classes hard to predict (low sensitivity)
- Model biased toward predicting nonDRNA
- DRNA especially difficult (only 2/3 true samples caught)

---

## File Locations

```
CMSC-435-Project/
├── Dataset/
│   ├── StartingData.txt              # Raw input data
│   ├── Unfinished/
│   │   ├── pfeatureSequenced.fa      # FASTA sequences
│   │   ├── trainingLabels.csv        # Separated labels
│   │   └── labeledData.csv           # Features + labels
│   └── full_normalized_BSMOTE.csv    # Final training data
├── Python-Scripts/
│   ├── SplitData.py
│   ├── GlueLabels.py
│   ├── Normalization.py
│   ├── Oversample.py
│   ├── CalculateMetrics.py
│   └── Discretization.py
├── pfeature_standalone/
│   ├── runPFeature.py
│   ├── pfeature_comp.py
│   └── tempfile_out                  # Temporary pfeature outputs
├── TempfilesAndOutput/
│   ├── predictions_output.csv        # Latest predictions
│   └── metrics_output.txt            # Performance metrics
└── Pipeline11-30.rmp                 # RapidMiner workflow
```

---

## Quick Reference: Data Transformation

1. **Raw sequence** (length ~200 amino acids)
   ```
   MLKQVEIFTDGSCLGNPGPGGYGAILRYRGREKTFSAGYTRTTNNRMELMAAIVALEALKEHCE...
   ```

2. **DPC features extracted** (400 features)
   ```
   DPC_AA: 0.3, DPC_AC: 0.0, DPC_AD: 0.6, ...
   ```

3. **Normalized** (z-scores)
   ```
   DPC_AA: -0.99, DPC_AC: -0.46, DPC_AD: 0.18, ...
   ```

4. **Class added**
   ```
   ..., class: nonDRNA
   ```

5. **Prediction made**
   ```
   confidence(nonDRNA): 0.548, prediction: nonDRNA
   ```

---

## Important Constants

- **Feature count:** 400 (DPC features)
- **Class count:** 4 (nonDRNA, RNA, DNA, DRNA)
- **Normalization method:** StandardScaler (z-score)
- **Oversampling method:** BorderlineSMOTE
- **Random state:** 42
- **Best model:** Random Forest, depth=200

---

## Notes for AI Analysis

1. **Dataset is highly imbalanced** - Most samples are nonDRNA
2. **Feature space is high-dimensional** - 400 features from sequence composition
3. **Oversampling helps but doesn't solve** minority class detection
4. **Model tends to over-predict** the majority class (nonDRNA)
5. **MCC is better metric than accuracy** due to class imbalance
6. **DRNA class is extremely rare** - only ~40 original samples
7. **Pipeline is modular** - Each step is a separate Python script callable by RapidMiner
8. **All paths are absolute** in the code (Windows format)
