import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from make_sample_data import make  # noqa: E402

from insurance_api.train import train  # noqa: E402


@pytest.fixture(scope="session")
def model_path(tmp_path_factory) -> Path:
    """Train once on synthetic data and share the artifact across tests."""
    tmp = tmp_path_factory.mktemp("artifacts")
    data = tmp / "sample.csv"
    make(3000).to_csv(data, index=False)
    out = tmp / "model.joblib"
    train(data, out, params={"n_estimators": 50})
    return out


@pytest.fixture
def customer() -> dict:
    return {
        "Gender": "Male",
        "Age": 44,
        "Driving_License": 1,
        "Region_Code": 28,
        "Previously_Insured": 0,
        "Vehicle_Age": "> 2 Years",
        "Vehicle_Damage": "Yes",
        "Annual_Premium": 40454.0,
        "Policy_Sales_Channel": 26,
        "Vintage": 217,
    }
