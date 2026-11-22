"""Feature definitions and the preprocessing pipeline, shared by training and serving."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

TARGET = "Response"

NUMERIC = ["Age", "Annual_Premium", "Vintage", "Driving_License", "Previously_Insured"]
CATEGORICAL = ["Gender", "Vehicle_Age", "Vehicle_Damage", "Region_Code", "Policy_Sales_Channel"]
FEATURES = NUMERIC + CATEGORICAL


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Make raw Kaggle data and API input look the same."""
    out = df[FEATURES].copy()
    # Region and sales channel are IDs, not quantities, so treat them as categories
    for col in ["Region_Code", "Policy_Sales_Channel"]:
        out[col] = out[col].astype(float).astype(int).astype(str)
    return out


def build_preprocessor() -> ColumnTransformer:
    # Rare regions/channels are grouped together, unseen ones map to that group too
    return ColumnTransformer(
        [
            ("num", "passthrough", NUMERIC),
            (
                "cat",
                OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.005, sparse_output=False),
                CATEGORICAL,
            ),
        ],
        verbose_feature_names_out=False,
    )


def source_column(transformed_name: str) -> str:
    """Map a one-hot column like 'Vehicle_Damage_Yes' back to 'Vehicle_Damage'."""
    for col in sorted(FEATURES, key=len, reverse=True):
        if transformed_name == col or transformed_name.startswith(col + "_"):
            return col
    return transformed_name
