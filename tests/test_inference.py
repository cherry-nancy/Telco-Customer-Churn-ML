import pandas as pd
import pytest

from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.serving import inference

VALID_OUTPUTS = {"Likely to churn", "Not likely to churn"}


def _serving_input(raw_df):
    return raw_df.drop(columns=["customerID", "Churn"])


def test_predict_returns_label(customer):
    assert inference.predict(customer) in VALID_OUTPUTS


def test_single_row_encoding_matches_batch(raw_df):
    # Regression test for the drop_first bug: encoding one row at a time used to
    # zero out every multi-category feature (Contract, InternetService, ...)
    df = _serving_input(raw_df)
    batch = inference._serve_transform(df).reset_index(drop=True)
    single = pd.concat(
        [inference._serve_transform(df.iloc[[i]]) for i in range(len(df))]
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(single, batch, check_dtype=False)


def test_serving_encoding_matches_training(raw_df):
    # Train/serve skew check: serving must build the same features training did
    served = inference._serve_transform(_serving_input(raw_df)).reset_index(drop=True)
    trained = build_features(preprocess_data(raw_df.copy()))[inference.FEATURE_COLS]
    pd.testing.assert_frame_equal(
        served.astype(float), trained.astype(float).reset_index(drop=True)
    )


@pytest.mark.parametrize("field,value", [
    ("Contract", "Two year"),
    ("InternetService", "DSL"),
    ("PaymentMethod", "Credit card (automatic)"),
])
def test_multi_category_fields_reach_the_model(customer, field, value):
    # Changing a multi-category field must change the encoded features
    before = inference._serve_transform(pd.DataFrame([customer]))
    after = inference._serve_transform(pd.DataFrame([{**customer, field: value}]))
    assert not before.equals(after)
