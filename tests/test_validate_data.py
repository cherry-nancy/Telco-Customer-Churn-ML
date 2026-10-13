from src.utils.validate_data import validate_telco_data


def test_valid_data_passes(raw_df):
    is_valid, failed = validate_telco_data(raw_df)
    assert is_valid
    assert failed == []


def test_unknown_gender_fails(raw_df):
    raw_df.loc[0, "gender"] = "X"
    is_valid, failed = validate_telco_data(raw_df)
    assert not is_valid
    assert "expect_column_values_to_be_in_set" in failed


def test_negative_tenure_fails(raw_df):
    raw_df.loc[0, "tenure"] = -5
    is_valid, failed = validate_telco_data(raw_df)
    assert not is_valid
    assert "expect_column_values_to_be_between" in failed


def test_missing_column_fails(raw_df):
    is_valid, failed = validate_telco_data(raw_df.drop(columns=["Contract"]))
    assert not is_valid
    assert "expect_column_to_exist" in failed
