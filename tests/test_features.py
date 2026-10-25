from pathlib import Path

from src.data.preprocess import preprocess_data
from src.features.build_features import build_features

FEATURE_COLS_FILE = (
    Path(__file__).resolve().parents[1]
    / "src/serving/model/3b1a41221fc44548aed629fa42b762e0/artifacts/feature_columns.txt"
)


def _features(raw_df):
    return build_features(preprocess_data(raw_df))


def test_preprocess_cleans_target_and_ids(raw_df):
    df = preprocess_data(raw_df)
    assert "customerID" not in df.columns
    assert set(df["Churn"].unique()) <= {0, 1} # <= means "is a subset of" so this checks that the Churn column only has 0s and 1s, no other values.
    # the fixture has a blank TotalCharges; it must become a number, not NaN
    assert df["TotalCharges"].notna().all() # True only if no row in TotalCharges is missing


def test_build_features_all_numeric_no_nans(raw_df):
    df = _features(raw_df)
    assert df.isna().sum().sum() == 0
    assert all(dtype.kind in "biuf" for dtype in df.dtypes), df.dtypes[df.dtypes == object]


def test_build_features_matches_served_model_schema(raw_df):
    # The serving model expects exactly these columns; training must keep producing them
    expected = [ln.strip() for ln in FEATURE_COLS_FILE.read_text().splitlines() if ln.strip()]
    produced = _features(raw_df).drop(columns=["Churn"]).columns
    assert set(produced) == set(expected)
