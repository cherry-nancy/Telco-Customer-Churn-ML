import great_expectations as gx
import pandas as pd
from typing import Tuple, List


def validate_telco_data(df) -> Tuple[bool, List[str]]:
    """
    Comprehensive data validation for Telco Customer Churn dataset using Great Expectations.
    
    This function implements critical data quality checks that must pass before model training.
    It validates data integrity, business logic constraints, and statistical properties
    that the ML model expects.
    
    """
    print("🔍 Starting data validation with Great Expectations...")
    
    # Raw TotalCharges is text with blank strings for brand-new customers;
    # coerce on a copy so numeric checks work (real cleaning happens in preprocess)
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Collect expectations into a suite (GX 1.x API; needs an active context first)
    context = gx.get_context(mode="ephemeral")
    suite = gx.ExpectationSuite(name="telco_churn_suite")
    
    # === SCHEMA VALIDATION - ESSENTIAL COLUMNS ===
    print("   📋 Validating schema and required columns...")
    
    # Customer identifier must exist (required for business operations)  
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="customerID"))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="customerID"))
    
    # Core demographic features
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="gender"))
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="Partner"))
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="Dependents"))
    
    # Service features (critical for churn analysis)
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="PhoneService"))
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="InternetService"))
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="Contract"))
    
    # Financial features (key churn predictors)
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="tenure"))
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="MonthlyCharges"))
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="TotalCharges"))
    
    # === BUSINESS LOGIC VALIDATION ===
    print("   💼 Validating business logic constraints...")
    
    # Gender must be one of expected values (data integrity)
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(column="gender", value_set=["Male", "Female"]))
    
    # Yes/No fields must have valid values
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(column="Partner", value_set=["Yes", "No"]))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(column="Dependents", value_set=["Yes", "No"]))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(column="PhoneService", value_set=["Yes", "No"]))
    
    # Contract types must be valid (business constraint)
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(
        column="Contract",
        value_set=["Month-to-month", "One year", "Two year"]
    ))
    
    # Internet service types (business constraint)
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(
        column="InternetService",
        value_set=["DSL", "Fiber optic", "No"]
    ))
    
    # === NUMERIC RANGE VALIDATION ===
    print("   📊 Validating numeric ranges and business constraints...")
    
    # Tenure must be non-negative (business logic - can't have negative tenure)
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="tenure", min_value=0))
    
    # Monthly charges must be positive (business logic - no free service)
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="MonthlyCharges", min_value=0))
    
    # Total charges should be non-negative (business logic)
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="TotalCharges", min_value=0))
    
    # === STATISTICAL VALIDATION ===
    print("   📈 Validating statistical properties...")
    
    # Tenure should be reasonable (max ~10 years = 120 months for telecom)
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="tenure", min_value=0, max_value=120))
    
    # Monthly charges should be within reasonable business range
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="MonthlyCharges", min_value=0, max_value=200))
    
    # No missing values in critical numeric features  
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="tenure"))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="MonthlyCharges"))
    
    # === DATA CONSISTENCY CHECKS ===
    print("   🔗 Validating data consistency...")
    
    # Total charges should generally be >= Monthly charges (except for very new customers)
    # This is a business logic check to catch data entry errors
    suite.add_expectation(gx.expectations.ExpectColumnPairValuesAToBeGreaterThanB(
        column_A="TotalCharges",
        column_B="MonthlyCharges",
        or_equal=True,
        mostly=0.95  # Allow 5% exceptions for edge cases
    ))
    
    # === RUN VALIDATION SUITE ===
    print("   ⚙️  Running complete validation suite...")
    batch = (
        context.data_sources.add_pandas("pandas")
        .add_dataframe_asset("telco")
        .add_batch_definition_whole_dataframe("batch")
        .get_batch(batch_parameters={"dataframe": df})
    )
    results = batch.validate(suite)
    
    # === PROCESS RESULTS ===
    # Extract failed expectations for detailed error reporting
    failed_expectations = []
    for r in results.results:
        if not r.success:
            failed_expectations.append(r.expectation_config.type)
    
    # Print validation summary
    total_checks = len(results.results)
    passed_checks = sum(1 for r in results.results if r.success)
    failed_checks = total_checks - passed_checks
    
    if results.success:
        print(f"✅ Data validation PASSED: {passed_checks}/{total_checks} checks successful")
    else:
        print(f"❌ Data validation FAILED: {failed_checks}/{total_checks} checks failed")
        print(f"   Failed expectations: {failed_expectations}")
    
    return results.success, failed_expectations
