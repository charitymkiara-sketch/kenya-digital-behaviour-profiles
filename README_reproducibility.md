# Supplementary Material 1: reproducibility files

This archive contains the analysis script, a separate sensitivity-analysis script excluding computer and bank-account ownership, pinned Python package versions, final analysis CSVs and the six current manuscript figures. The Afrobarometer microdata are not included.

## Data and use conditions

Download the Kenya Round 10 (2024) public-use data from Afrobarometer: https://www.afrobarometer.org/survey-resource/kenya-round-10-data-2024/ . Follow Afrobarometer's data-use and citation policy: https://www.afrobarometer.org/data/data-usage-and-access-policy/ .

## Reproduce the primary analysis

Use Python 3.12.14, install the pinned dependencies, and run:

```bash
python -m pip install -r requirements.txt
python reproduce_digital_profiles.py --data "KEN_R10.Data_28June24.wtd_.final_.release_updated.13Feb25.sav" --output analysis_output
```

The primary script creates the clustering assignments, cluster diagnostics, stability and clustering sensitivity outputs, nested cross-validation results, permutation importance, and holdout results.

## Reproduce the model sensitivity analysis

After the primary script creates the assignment file, run:

```bash
python sensitivity_no_computer_bank.py --data "KEN_R10.Data_28June24.wtd_.final_.release_updated.13Feb25.sav" --assignments analysis_output/profile_assignments.csv --output analysis_output/sensitivity_no_computer_bank
```

This refits the supervised models after removing Q90D (computer ownership) and Q90E (bank-account ownership). The figure PNGs and CSV files in this archive are the outputs used for the revised manuscript.

## Settings

- K-means: k = 2–6, 50 initialisations, random seed 42.
- Stability: 50 random 80% subsamples.
- K-prototypes sensitivity: k = 3, Huang initialisation, 30 initialisations, 100 iterations, gamma = 0.5, seed 42.
- Prediction: stratified 80:20 development/holdout split; nested 5-fold/3-fold cross-validation is conducted on the development sample only; the holdout is used once.
