#!/usr/bin/env python3
"""Repeat supervised prediction after excluding computer and bank-account ownership."""
import argparse
import importlib.util
from pathlib import Path
import pandas as pd
import pyreadstat

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("digital_profiles", HERE / "reproduce_digital_profiles.py")
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True, help="Afrobarometer Kenya Round 10 .sav file")
parser.add_argument("--assignments", default="analysis_output/profile_assignments.csv")
parser.add_argument("--output", default="analysis_output/sensitivity_no_computer_bank")
args = parser.parse_args()
raw, _ = pyreadstat.read_sav(args.data, apply_value_formats=False)
d = model.clean_data(raw)
y = pd.read_csv(args.assignments)["high_engagement"].to_numpy()
model.PREDICTOR_CATEGORICAL = [c for c in model.PREDICTOR_CATEGORICAL if c not in ["Q90D", "Q90E"]]
model.PREDICTORS = model.PREDICTOR_NUMERIC + model.PREDICTOR_CATEGORICAL
out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
model.prediction_analysis(d, y, out)
print(f"Sensitivity outputs written to {out}")
