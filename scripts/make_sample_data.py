"""Generate a small synthetic dataset with the same columns as the Kaggle data.

Used by tests and CI so the pipeline runs without downloading anything.
The numbers are made up; real results come from the Kaggle train.csv.

Usage:
    python scripts/make_sample_data.py --rows 5000 --out data/sample.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make(rows: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(20, 80, rows)
    prev = rng.binomial(1, 0.46, rows)
    damage = np.where(rng.random(rows) < np.where(prev == 1, 0.05, 0.9), "Yes", "No")
    vehicle_age = np.where(age < 30, "< 1 Year", np.where(rng.random(rows) < 0.9, "1-2 Year", "> 2 Years"))

    # Interest is driven mainly by: not insured yet, had vehicle damage, middle age
    logit = -4.8 + 2.2 * (prev == 0) + 1.5 * (damage == "Yes") + 0.6 * ((age > 30) & (age < 55))
    response = rng.binomial(1, 1 / (1 + np.exp(-logit)))

    return pd.DataFrame(
        {
            "id": np.arange(1, rows + 1),
            "Gender": rng.choice(["Male", "Female"], rows, p=[0.54, 0.46]),
            "Age": age,
            "Driving_License": rng.binomial(1, 0.998, rows),
            "Region_Code": rng.choice([28, 8, 46, 41, 15, 30, 29, 50, 3, 11], rows).astype(float),
            "Previously_Insured": prev,
            "Vehicle_Age": vehicle_age,
            "Vehicle_Damage": damage,
            "Annual_Premium": np.round(rng.lognormal(10.3, 0.4, rows), 0),
            "Policy_Sales_Channel": rng.choice([152, 26, 124, 160, 156, 122], rows).astype(float),
            "Vintage": rng.integers(10, 300, rows),
            "Response": response,
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--out", type=Path, default=Path("data/sample.csv"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    make(args.rows).to_csv(args.out, index=False)
    print(f"Wrote {args.rows} rows to {args.out}")
