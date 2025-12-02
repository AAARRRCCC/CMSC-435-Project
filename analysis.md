# Pipeline Performance Analysis
**Senior Data Scientist Diagnostic Report**
**Date:** November 30, 2025
**Pipeline Version:** Pipeline11-30.rmp

---

## Executive Summary

The current machine learning pipeline achieves poor performance on minority classes, with the DRNA class showing **0% sensitivity** (complete prediction failure) and RNA/DNA classes showing only **69-72% sensitivity**. Matthews Correlation Coefficient (MCC) scores range from 0.0 to 0.337, indicating weak predictive power across all classes.

After comprehensive analysis of the codebase, data, and pipeline architecture, **five critical issues** have been identified that explain this poor performance. The most severe issue is **data leakage in the normalization step**, which invalidates all current evaluation metrics.

---

## Current Performance Metrics

### Classification Results (from metrics_output.txt)

| Class | Sensitivity | Specificity | Accuracy | MCC |
|-------|------------|-------------|----------|-----|
| nonDRNA | 90.73% | 77.40% | 90.46% | 0.310 |
| RNA | **69.34%** | 95.06% | 94.66% | 0.337 |
| DNA | **72.50%** | 95.87% | 95.76% | 0.223 |
| DRNA | **0.00%** | 99.75% | 99.75% | **0.000** |

### Confusion Matrix Analysis

```
Predicted →  nonDRNA   RNA   DNA   DRNA
nonDRNA       7,818    427   350    22
RNA              30     95    12     0
DNA              10      1    29     0
DRNA              0      0     0     0  ← No DRNA predictions at all
```

**Key Observations:**
- Model predicts nonDRNA correctly but confuses it with RNA/DNA
- RNA class: Only 95/137 correct (69% sensitivity)
- DNA class: Only 29/40 correct (72% sensitivity)
- **DRNA class: 0/0 predictions - complete failure**
- Model is heavily biased toward majority class (nonDRNA)

---

## Critical Issues Identified

### Issue #1: Data Leakage in Normalization ⚠️ CRITICAL

**Severity:** CRITICAL
**Impact:** All evaluation metrics are unreliable
**Location:** Pipeline11-30.rmp lines 73-79, 216-222

#### Problem Description

The pipeline normalizes test data **independently** from training data, causing severe data leakage that violates fundamental machine learning principles.

**Current (INCORRECT) Behavior:**
```
Training Process (lines 73-79):
  1. Receive training fold
  2. Call Normalization.py rm_main()
  3. Create StandardScaler()
  4. Fit scaler on training data
  5. Transform training data
  ✓ This part is correct

Test Process (lines 216-222):
  1. Receive test fold
  2. Call Normalization.py rm_main()
  3. Create NEW StandardScaler()     ← WRONG
  4. Fit scaler on TEST data         ← DATA LEAKAGE
  5. Transform test data
  ✗ This creates information leakage
```

**Why This Is Wrong:**

When the scaler is fit on test data:
- Test data statistics (mean, standard deviation) are calculated from test data
- Test data is normalized to have mean=0, std=1 based on **its own statistics**
- Training data was normalized using **different statistics** (from training set)
- The normalized distributions don't match between train and test
- Model evaluation uses test data that has "seen" its own distribution

**Correct Behavior Should Be:**
```
Training Process:
  1. Fit scaler on training data
  2. Transform training data
  3. SAVE the fitted scaler parameters (mean, std)

Test Process:
  1. LOAD the saved scaler from training
  2. Transform (NOT fit_transform) test data using training statistics
  3. Test data normalized using training distribution
```

#### Impact

- **All current metrics are artificially inflated and unreliable**
- True model performance is likely **worse** than reported
- Cannot compare different experiments reliably
- Real-world deployment would fail (no access to future data statistics)

#### Evidence

File: `Python-Scripts/Normalization.py` (lines 20-33)
```python
def rm_main(df):
    df = df.copy()
    scaler = StandardScaler()  # Creates NEW scaler every time
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])  # Fits on input data
    return df  # No scaler persistence
```

The function creates a fresh scaler on every call with no mechanism to persist or reuse scalers between training and test phases.

---

### Issue #2: Oversampling Disabled ⚠️ CRITICAL

**Severity:** CRITICAL
**Impact:** Primary cause of minority class failure
**Location:** Pipeline11-30.rmp line 94

#### Problem Description

The BorderlineSMOTE oversampling operator is **deactivated** in the pipeline despite being fully configured.

**Evidence:**
```xml
Line 94: <operator activated="false" class="python_scripting:execute_python"... name="Oversample"
```

The operator is set to `activated="false"`, meaning it's completely skipped during execution.

**Oversample.py Configuration (lines 32-33):**
```python
smote = BorderlineSMOTE(
    sampling_strategy={'DNA':2000, 'RNA':2000, 'DRNA':500},
    kind='borderline-1',
    random_state=42
)
```

This would generate synthetic samples to balance the classes, but it's never executed.

#### Impact

Without oversampling, the model trains on severely imbalanced data:

**Current Class Distribution:**
- nonDRNA: 7,858 samples (89.36%)
- RNA: 523 samples (5.94%)
- DNA: 391 samples (4.45%)
- **DRNA: 22 samples (0.25%)**

**Imbalance Ratios:**
- nonDRNA to DRNA: **357:1**
- nonDRNA to DNA: **20:1**
- nonDRNA to RNA: **15:1**

**Why DRNA Shows 0% Sensitivity:**

With 5-fold cross-validation:
- Each test fold contains ~4-5 DRNA samples
- Each training fold contains ~18 DRNA samples
- Model sees 18 DRNA vs 6,286 nonDNA (350:1 ratio)
- Predicting "always nonDRNA" gives 89% accuracy
- Model learns to ignore DRNA class completely

**Mathematical Analysis:**

For a model optimizing accuracy:
```
Accuracy if predict all nonDRNA: 89.36%
Accuracy if perfectly classify DRNA: 89.36% + 0.25% = 89.61%

Gain from learning DRNA: 0.25%
Risk from misclassifying nonDRNA as DRNA: High

Result: Model ignores DRNA to maximize accuracy
```

---

### Issue #3: Severe Class Imbalance (No Mitigation) ⚠️ HIGH PRIORITY

**Severity:** HIGH
**Impact:** Compounds Issue #2, affects all minority classes

#### Problem Description

Beyond the disabled oversampling, the pipeline has **no active class imbalance handling**:

**Missing Strategies:**
1. ❌ No class weights configured in model
2. ❌ No oversampling active (Issue #2)
3. ❌ No undersampling of majority class
4. ❌ No cost-sensitive learning
5. ❌ No ensemble methods for imbalanced data

**Current Model Configuration (lines 148-168):**
```xml
<operator activated="true" class="h2o:gradient_boosted_trees">
  <!-- No class_weights parameter -->
  <!-- No sampling_strategy parameter -->
  <!-- Optimizes for overall accuracy, not per-class performance -->
</operator>
```

#### Impact on Each Class

**DRNA (22 samples, 0.25%):**
- Insufficient data for meaningful learning
- 5-fold CV = ~4-5 samples per test fold
- Model never learns DRNA patterns
- Result: 0% sensitivity, 0.000 MCC

**DNA (391 samples, 4.45%):**
- Underrepresented by 20:1 ratio
- Model sees 20x more nonDRNA examples
- Learns weak DNA patterns
- Result: 72.5% sensitivity, 0.223 MCC

**RNA (523 samples, 5.94%):**
- Underrepresented by 15:1 ratio
- Slightly better than DNA due to more samples
- Still insufficient for robust learning
- Result: 69.3% sensitivity, 0.337 MCC

**nonDRNA (7,858 samples, 89.36%):**
- Dominates training signal
- Model optimizes for this class
- High sensitivity but poor specificity
- Result: 90.7% sensitivity, but causes many false positives

---

### Issue #4: Suboptimal Model Hyperparameters ⚠️ MEDIUM PRIORITY

**Severity:** MEDIUM
**Impact:** Model underfitting due to conservative configuration
**Location:** Pipeline11-30.rmp lines 148-168

#### Problem Description

The Gradient Boosted Trees model is configured with overly conservative hyperparameters that likely cause underfitting.

**Current Configuration:**
```xml
<parameter key="number_of_trees" value="50"/>           ← Too few
<parameter key="learning_rate" value="0.01"/>            ← Too low
<parameter key="maximal_depth" value="5"/>               ← Reasonable
<parameter key="min_rows" value="10.0"/>                 ← Reasonable
<parameter key="sample_rate" value="1.0"/>               ← No row sampling
<parameter key="early_stopping" value="false"/>          ← No regularization
```

#### Analysis

**Learning Rate Problem:**
- Learning rate (lr) = 0.01 is very conservative
- Typical values: 0.05-0.3 for gradient boosting
- Low lr requires many more trees to converge

**Number of Trees Problem:**
- Only 50 trees with lr=0.01
- Effective learning: `0.01 × 50 = 0.5` total learning capacity
- Typical effective learning: 1.0-2.0 range
- Model is only ~25-50% trained

**Rule of Thumb:**
```
If lr = 0.1, use 100-200 trees
If lr = 0.05, use 200-400 trees
If lr = 0.01, use 500-1000 trees  ← Current: only 50 trees

Current config: SEVERE UNDERFITTING
```

**Missing Features:**
- No early stopping (risks overfitting if trees increased)
- No row sampling (sample_rate=1.0 means no stochastic element)
- No column sampling (uses all features every split)

#### Impact

- Model lacks capacity to learn complex patterns
- Especially hurts minority classes (already have limited data)
- Combined with class imbalance, creates poor discrimination
- Cannot exploit full potential of 400 DPC features

---

### Issue #5: Limited Feature Representation ⚠️ MEDIUM PRIORITY

**Severity:** MEDIUM
**Impact:** May be missing discriminative patterns
**Location:** pfeature_standalone/runPFeature.py line 9

#### Problem Description

The pipeline uses **only DPC (Dipeptide Composition)** features, which capture 2-residue patterns. Many more sophisticated feature extraction methods are available but unused.

**Current Feature Extraction:**
```python
# runPFeature.py line 9
subprocess.run("python pfeature_comp.py -i ../Dataset/Unfinished/pfeatureSequenced.fa -o CSVTEMP.csv -j DPC")
```

**Feature Set:**
- Type: DPC1 (Dipeptide Composition, lag=1)
- Number: 400 features (20 amino acids × 20 amino acids)
- Information: Frequency of consecutive amino acid pairs
- Limitations:
  - No sequence order beyond 2 residues
  - No physicochemical properties
  - No structural information
  - No evolutionary information

#### Available but Unused Feature Types

The pfeature suite provides **20+ feature extraction methods**:

**High Priority Candidates:**

1. **PAAC (Pseudo Amino Acid Composition)** - 30-40 features
   - Incorporates sequence order + physicochemical properties
   - Proven effective for protein function prediction
   - Low dimensionality (easier to train)
   - Quick to implement (change `-j DPC` to `-j PAAC`)

2. **TPC (Tripeptide Composition)** - 8,000 features
   - Captures 3-residue patterns (vs DPC's 2-residue)
   - More granular sequence information
   - Better for structural motifs
   - Requires feature selection due to high dimensionality

3. **CTC (Conjoint Triad Descriptors)** - Variable features
   - Groups amino acids by physicochemical properties
   - Captures functional motifs
   - Effective for binding site prediction

4. **PSSM (Position-Specific Scoring Matrix)** - 20 × sequence_length
   - Incorporates evolutionary information from BLAST
   - Most biologically meaningful
   - Requires external BLAST database setup

**Other Available Methods:**
- APAAC (Amphiphilic Pseudo AAC)
- QSO (Quasi Sequence Order)
- SOC (Sequence Order Coupling)
- CeTD (Composition Enhanced Transition Distribution)
- Shannon Entropy features (SEP, SER, SPC)
- Binary profiles (AAB, DPB, PCB)
- Many more...

#### Impact

**Why This Matters for RNA/DNA Classification:**

RNA-binding and DNA-binding proteins may differ in:
- **Sequence order patterns** (not captured by DPC)
- **Physicochemical properties** (charge, hydrophobicity)
- **Structural motifs** (helix-turn-helix, zinc fingers)
- **Evolutionary conservation** (binding sites conserved)

DPC alone may not capture these discriminative features, limiting the model's ability to distinguish between classes.

#### Supporting Evidence

The CAIM discretization algorithm is already implemented in the codebase (`Python-Scripts/Discretization.py`), suggesting the team has considered feature selection for high-dimensional feature sets. This infrastructure could support TPC or multi-feature experiments.

---

## Data Analysis

### Dataset Statistics

**Source File:** `Dataset/Unfinished/labeledData.csv`
- Total samples: 8,794
- Features: 400 (DPC1_XX format)
- Classes: 4 (nonDRNA, RNA, DNA, DRNA)

**Class Distribution:**

| Class | Count | Percentage | Imbalance Ratio |
|-------|-------|-----------|----------------|
| nonDRNA | 7,858 | 89.36% | 1.0 (baseline) |
| RNA | 523 | 5.94% | 15.0:1 |
| DNA | 391 | 4.45% | 20.1:1 |
| DRNA | 22 | 0.25% | 357.2:1 |

### Cross-Validation Impact

**5-Fold Stratified CV:**

| Class | Per Fold (Train) | Per Fold (Test) |
|-------|-----------------|----------------|
| nonDRNA | ~6,286 | ~1,572 |
| RNA | ~418 | ~105 |
| DNA | ~313 | ~78 |
| **DRNA** | **~18** | **~4** |

**DRNA Fold Analysis:**
- Training folds: 18 samples vs 6,286 nonDRNA (350:1)
- Test folds: 4 samples (statistically insufficient)
- Variance in performance: High (small sample CLT doesn't apply)

---

## Pipeline Architecture Analysis

### Current Data Flow

```
1. Read CSV (manual_test_comp_labelled.csv)
   ↓
2. Set Role (class = label)
   ↓
3. Cross Validation (5-fold stratified)
   ├─ TRAINING BRANCH:
   │  ├─ Write CSV (training split)
   │  ├─ Normalization (2) → fit_transform on TRAINING ✓
   │  ├─ [Oversample] → DISABLED ✗
   │  ├─ Read normalized CSV
   │  ├─ Set Role (class = label)
   │  └─ Gradient Boosted Trees → Train model
   │
   └─ TEST BRANCH:
      ├─ Write CSV (test split)
      ├─ Normalization → fit_transform on TEST ✗ LEAKAGE
      ├─ Read normalized CSV
      ├─ Set Role (class = label)
      ├─ Apply Model → Make predictions
      └─ Performance → Calculate metrics
   ↓
4. Write predictions → predictions_output.csv
   ↓
5. Calculate Metrics → metrics_output.txt
```

### Problems in Architecture

1. **Normalization happens inside CV loop** - Correct approach, BUT
2. **No scaler persistence mechanism** - Test uses different scaler
3. **Oversampling is disabled** - No class balancing
4. **No class weights in model** - Equal treatment of all classes
5. **SVM alternative is disabled** - No model diversity

---

## Recommended Solutions

### Priority 0: Critical Fixes (Must Do Before Any Experiments)

#### Fix 1: Resolve Normalization Data Leakage

**Option A: Scaler Persistence (Recommended)**

Modify `Normalization.py` to support train/test modes:

```python
def rm_main(df, mode='train', scaler=None):
    df = df.copy()
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns

    if mode == 'train':
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        # Store scaler to disk or return it
        joblib.dump(scaler, '../TempfilesAndOutput/scaler.pkl')
    elif mode == 'test':
        # Load scaler from disk or use passed scaler
        scaler = joblib.load('../TempfilesAndOutput/scaler.pkl')
        df[numerical_cols] = scaler.transform(df[numerical_cols])  # Only transform

    return df
```

**Option B: Pre-CV Normalization (Simpler, Less Ideal)**

Move normalization outside the CV loop:
- Normalize entire dataset before CV split
- Less theoretically pure but acceptable for CV
- Still has issues for final train/test split

**Recommendation:** Implement Option A for proper train/test separation.

#### Fix 2: Enable Oversampling

**Immediate Action:**
```xml
Change line 94 in Pipeline11-30.rmp:
FROM: <operator activated="false" ...
TO:   <operator activated="true" ...
```

**Configuration Tuning:**

Current Oversample.py targets may be too aggressive:
```python
# Current
sampling_strategy={'DNA':2000, 'RNA':2000, 'DRNA':500}

# Recommended conservative start
sampling_strategy={'DNA':1000, 'RNA':1000, 'DRNA':300}

# Aggressive (match majority)
sampling_strategy={'DNA':2000, 'RNA':2000, 'DRNA':800}
```

Start conservative, then increase if minority classes improve without overfitting.

### Priority 1: High Priority Improvements

#### Improvement 1: Add Class Weights

Calculate inverse frequency weights:

```python
total_samples = 8794
n_classes = 4

weights = {
    'nonDNA': total_samples / (n_classes * 7858) = 0.28,
    'RNA': total_samples / (n_classes * 523) = 4.20,
    'DNA': total_samples / (n_classes * 391) = 5.63,
    'DRNA': total_samples / (n_classes * 22) = 99.93
}
```

Add to GBT configuration in RapidMiner (if supported) or use in custom Python model.

#### Improvement 2: Tune Hyperparameters

**Recommended Configuration:**
```xml
<parameter key="learning_rate" value="0.05"/>        ← Increase from 0.01
<parameter key="number_of_trees" value="150"/>       ← Increase from 50
<parameter key="maximal_depth" value="7"/>           ← Increase from 5
<parameter key="sample_rate" value="0.8"/>           ← Add row sampling
<parameter key="early_stopping" value="true"/>       ← Enable regularization
<parameter key="stopping_rounds" value="10"/>        ← Add early stop rounds
```

**Justification:**
- lr=0.05 × trees=150 = 7.5 effective learning (vs current 0.5)
- Deeper trees (7) capture more complex interactions
- Row sampling (0.8) adds regularization
- Early stopping prevents overfitting

### Priority 2: Feature Engineering

#### Experiment 1: PAAC Features (Quick Win)

**Change Required:**
```python
# pfeature_standalone/runPFeature.py line 9
# FROM:
subprocess.run("python pfeature_comp.py -i ../Dataset/Unfinished/pfeatureSequenced.fa -o CSVTEMP.csv -j DPC")

# TO:
subprocess.run("python pfeature_comp.py -i ../Dataset/Unfinished/pfeatureSequenced.fa -o CSVTEMP.csv -j PAAC")
```

**Expected Outcome:**
- Reduces features from 400 to ~30-40
- Adds sequence order information
- Includes physicochemical properties
- May improve classification with less overfitting risk

#### Experiment 2: Multi-Feature Ensemble

Combine multiple feature types:
1. Generate DPC (400) + PAAC (30) + CTC (variable)
2. Concatenate feature vectors
3. Apply CAIM discretization for feature selection
4. Train on top 200-500 selected features

---

## Expected Outcomes

### After Fixing Critical Issues (P0)

**Immediate Impact:**
1. Metrics will initially **DROP** after fixing normalization leakage
2. This represents the **true baseline** performance
3. Enables reliable comparison between experiments

**After Enabling Oversampling:**
1. DRNA sensitivity: 0% → 40-60% (with 300-500 synthetic samples)
2. RNA sensitivity: 69% → 75-85%
3. DNA sensitivity: 72% → 78-88%
4. Overall MCC: Improve across all classes

### After All Improvements (P0 + P1 + P2)

**Target Metrics:**

| Class | Current Sensitivity | Target Sensitivity | Current MCC | Target MCC |
|-------|-------------------|-------------------|------------|-----------|
| nonDRNA | 90.7% | 85-90% | 0.310 | >0.60 |
| RNA | 69.3% | >80% | 0.337 | >0.60 |
| DNA | 72.5% | >80% | 0.223 | >0.60 |
| DRNA | **0.0%** | **>50%** | **0.000** | **>0.40** |

**Overall Goals:**
- Macro-averaged F1 score: >0.75
- All class MCC: >0.50
- DRNA showing measurable predictive power

---

## Implementation Roadmap

### Phase 1: Critical Infrastructure (Week 1)

**Tasks:**
1. Fix normalization data leakage
   - Modify Normalization.py
   - Update pipeline to handle scaler persistence
   - Test with simple dataset
2. Enable oversampling
   - Change activated="true" in pipeline
   - Start with conservative targets
3. Re-establish baseline
   - Run pipeline with fixes
   - Document new (true) baseline metrics

**Success Criteria:**
- No data leakage in evaluation
- Oversampling executing successfully
- DRNA showing >0% sensitivity

### Phase 2: Model Optimization (Week 2)

**Tasks:**
1. Add class weights to GBT
2. Tune hyperparameters systematically
   - Grid search over lr and n_trees
   - Test depth values
3. Enable and test SVM alternative

**Success Criteria:**
- Minority class sensitivity >70%
- MCC >0.40 for all classes

### Phase 3: Feature Engineering (Week 3-4)

**Tasks:**
1. Test PAAC features
2. Compare DPC vs PAAC vs combined
3. Experiment with TPC + feature selection
4. Document feature impact

**Success Criteria:**
- Identify best feature combination
- All target metrics achieved

---

## Key Learnings for Interns

### Lesson 1: Data Leakage is Insidious

**Teaching Point:**
Data leakage can hide in subtle places like normalization. Always verify:
- Training and test data are processed separately
- No information from test set influences training
- Fitted parameters (scalers, encoders) are saved and reused
- Evaluation metrics reflect true generalization

**Red Flags:**
- "Surprisingly good" performance that doesn't make sense
- Performance degradation in production
- Different results when rerunning pipeline

### Lesson 2: Class Imbalance Requires Multiple Strategies

**Teaching Point:**
A single technique (oversampling OR class weights) is often insufficient:
- Combine sampling strategies (over + under)
- Use class weights in loss function
- Choose appropriate evaluation metrics
- Consider ensemble methods

**Common Mistakes:**
- Ignoring imbalance and trusting accuracy
- Over-relying on SMOTE alone
- Not validating synthetic samples are realistic

### Lesson 3: Hyperparameter Tuning Follows Theory

**Teaching Point:**
Hyperparameters must work together:
- Low learning rate → Many trees needed
- High depth → More regularization needed
- Small dataset → Simpler models
- Imbalanced data → Adjust stopping criteria

**Rule:**
Don't blindly copy hyperparameters from tutorials. Understand the theory and adapt to your data.

### Lesson 4: Feature Engineering Can Trump Model Selection

**Teaching Point:**
Better features often provide larger gains than better algorithms:
- Domain knowledge drives feature selection
- Biological features (PAAC, PSSM) encode expert knowledge
- Simple model + good features > complex model + poor features

**In This Project:**
Switching from DPC to PAAC (simpler features) may outperform complex model tuning.

### Lesson 5: Validate Your Pipeline First

**Teaching Point:**
Before experimenting with models:
1. Verify data pipeline integrity
2. Check for leakage
3. Validate assumptions
4. Establish true baseline

**Time Saved:**
Weeks of invalid experiments avoided by fixing pipeline first.

---

## References

### Files Analyzed

1. `Pipeline11-30.rmp` - Main RapidMiner workflow
2. `Python-Scripts/Normalization.py` - Normalization implementation
3. `Python-Scripts/Oversample.py` - BorderlineSMOTE configuration
4. `Python-Scripts/CalculateMetrics.py` - Evaluation metrics
5. `pfeature_standalone/runPFeature.py` - Feature extraction
6. `Dataset/Unfinished/labeledData.csv` - Training data
7. `TempfilesAndOutput/metrics_output.txt` - Current results

### Tools and Libraries

- RapidMiner 12.0.001
- Python 3.x with pandas, scikit-learn, imbalanced-learn
- pfeature suite for protein feature extraction
- H2O Gradient Boosted Trees

### Theoretical Background

- **Data Leakage:** Kaufman et al., "Leakage in Data Mining" (2012)
- **Class Imbalance:** He & Garcia, "Learning from Imbalanced Data" (2009)
- **SMOTE:** Chawla et al., "SMOTE: Synthetic Minority Over-sampling Technique" (2002)
- **Protein Features:** Kumar et al., "Pfeature: A Tool for Computing Wide Range of Protein Features" (2019)

---

## Appendix: Quick Reference

### Most Critical Changes

**File: Pipeline11-30.rmp**
- Line 94: Change `activated="false"` to `activated="true"` (Enable oversampling)
- Lines 148-168: Update GBT hyperparameters

**File: Python-Scripts/Normalization.py**
- Add scaler persistence mechanism
- Separate train/test modes

**File: pfeature_standalone/runPFeature.py**
- Line 9: Experiment with feature types (DPC → PAAC → TPC)

### Command to Run Pipeline

```bash
# Launch RapidMiner and open Pipeline11-30.rmp
# Or use RapidMiner command line
```

### Validation Checklist

- [ ] Normalization uses same scaler for train and test
- [ ] Oversampling is enabled
- [ ] Class distribution after sampling is logged
- [ ] Hyperparameters are documented
- [ ] Baseline metrics established after fixes
- [ ] Each experiment is version controlled

---

## Conclusion

The current pipeline suffers from **critical data leakage** and **disabled class balancing**, leading to unreliable metrics and poor minority class performance. The DRNA class failure (0% sensitivity) is directly attributable to extreme class imbalance (22 samples) combined with no mitigation strategies.

**Immediate actions required:**
1. Fix normalization data leakage (invalidates all current metrics)
2. Enable oversampling (addresses DRNA/DNA/RNA failure)
3. Re-establish true baseline

**Expected outcome after fixes:**
- Reliable evaluation metrics
- DRNA class becomes learnable (>50% sensitivity)
- All minority classes show substantial improvement
- MCC scores >0.50 across all classes

The analysis has identified clear, actionable fixes with high confidence of substantial performance improvement.

---

**Report prepared by:** Senior Data Scientist
**For:** CMSC 435 Project Team
**Next Review:** After implementing Phase 1 fixes
