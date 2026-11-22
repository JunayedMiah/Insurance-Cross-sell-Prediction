import pandas as pd

from insurance_api.features import FEATURES, clean, source_column


def test_source_column_maps_one_hot_names():
    assert source_column("Vehicle_Damage_Yes") == "Vehicle_Damage"
    assert source_column("Vehicle_Age_< 1 Year") == "Vehicle_Age"
    assert source_column("Region_Code_28") == "Region_Code"
    assert source_column("Age") == "Age"


def test_clean_treats_ids_as_categories():
    df = pd.DataFrame([{f: 0 for f in FEATURES} | {"Region_Code": 28.0, "Policy_Sales_Channel": 26.0}])
    out = clean(df)
    assert out.loc[0, "Region_Code"] == "28"
    assert out.loc[0, "Policy_Sales_Channel"] == "26"
    assert list(out.columns) == FEATURES
