"""Tests for Tax-Calculator validator."""

import pytest

from cosilico_validators.validators.base import TestCase, ValidatorType
from cosilico_validators.validators.taxcalc import (
    MARS_MAPPING,
    SUPPORTED_VARIABLES,
    VARIABLE_MAPPING,
    TaxCalculatorValidator,
)

# Check if taxcalc is available
try:
    import taxcalc  # noqa: F401

    HAS_TAXCALC = True
except ImportError:
    HAS_TAXCALC = False

# Skip integration tests if taxcalc is not installed
needs_taxcalc = pytest.mark.skipif(not HAS_TAXCALC, reason="taxcalc not installed")


class TestTaxCalculatorValidatorMetadata:
    """Tests for validator metadata and configuration."""

    def test_validator_name(self):
        validator = TaxCalculatorValidator()
        assert validator.name == "Tax-Calculator"

    def test_validator_type(self):
        validator = TaxCalculatorValidator()
        assert validator.validator_type == ValidatorType.SUPPLEMENTARY

    def test_supported_variables_include_common_names(self):
        assert "eitc" in SUPPORTED_VARIABLES
        assert "income_tax" in SUPPORTED_VARIABLES
        assert "ctc" in SUPPORTED_VARIABLES
        assert "standard_deduction" in SUPPORTED_VARIABLES
        assert "agi" in SUPPORTED_VARIABLES
        assert "taxable_income" in SUPPORTED_VARIABLES

    def test_supports_variable(self):
        validator = TaxCalculatorValidator()
        assert validator.supports_variable("eitc")
        assert validator.supports_variable("EITC")  # Case insensitive
        assert validator.supports_variable("income_tax")
        assert validator.supports_variable("ctc")
        assert not validator.supports_variable("nonexistent_variable")


class TestMarsMapping:
    """Tests for filing status mapping."""

    def test_single_status(self):
        assert MARS_MAPPING["SINGLE"] == 1

    def test_joint_status(self):
        assert MARS_MAPPING["JOINT"] == 2
        assert MARS_MAPPING["MARRIED_FILING_JOINTLY"] == 2

    def test_separate_status(self):
        assert MARS_MAPPING["MARRIED_FILING_SEPARATELY"] == 3
        assert MARS_MAPPING["SEPARATE"] == 3

    def test_head_of_household_status(self):
        assert MARS_MAPPING["HEAD_OF_HOUSEHOLD"] == 4
        assert MARS_MAPPING["HOUSEHOLD_HEAD"] == 4

    def test_widow_status(self):
        assert MARS_MAPPING["WIDOW"] == 5
        assert MARS_MAPPING["WIDOWER"] == 5


class TestVariableMapping:
    """Tests for output variable mapping."""

    def test_eitc_mapping(self):
        assert VARIABLE_MAPPING["eitc"] == "eitc"
        assert VARIABLE_MAPPING["earned_income_credit"] == "eitc"

    def test_income_tax_mapping(self):
        assert VARIABLE_MAPPING["income_tax"] == "iitax"
        assert VARIABLE_MAPPING["federal_income_tax"] == "iitax"

    def test_ctc_mapping(self):
        assert VARIABLE_MAPPING["ctc"] == "c07220"
        assert VARIABLE_MAPPING["child_tax_credit"] == "c07220"

    def test_standard_deduction_mapping(self):
        assert VARIABLE_MAPPING["standard_deduction"] == "standard"

    def test_agi_mapping(self):
        assert VARIABLE_MAPPING["agi"] == "c00100"
        assert VARIABLE_MAPPING["adjusted_gross_income"] == "c00100"

    def test_taxable_income_mapping(self):
        assert VARIABLE_MAPPING["taxable_income"] == "c04800"


@needs_taxcalc
class TestTaxCalculatorValidation:
    """Integration tests for Tax-Calculator validation.

    These tests require taxcalc to be installed and run actual calculations.
    """

    @pytest.fixture
    def validator(self):
        return TaxCalculatorValidator()

    def test_single_filer_no_income(self, validator):
        """Single filer with zero income should have zero tax."""
        test_case = TestCase(
            name="Zero income",
            inputs={"earned_income": 0, "filing_status": "SINGLE"},
            expected={"income_tax": 0},
        )
        result = validator.validate(test_case, "income_tax", year=2023)
        assert result.success
        assert result.calculated_value == 0

    def test_single_filer_with_wages(self, validator):
        """Single filer with wages should have positive AGI."""
        test_case = TestCase(
            name="Wages only",
            inputs={"earned_income": 50000, "filing_status": "SINGLE", "age": 35},
            expected={"agi": 50000},
        )
        result = validator.validate(test_case, "agi", year=2023)
        assert result.success
        assert result.calculated_value is not None
        # AGI should be close to wages (may differ slightly due to adjustments)
        assert abs(result.calculated_value - 50000) < 1000

    def test_eitc_eligibility(self, validator):
        """Low income worker with children should receive EITC."""
        test_case = TestCase(
            name="EITC eligible",
            inputs={
                "earned_income": 20000,
                "filing_status": "SINGLE",
                "num_children": 2,
                "age": 30,
            },
            expected={"eitc": 5000},  # Approximate expected range
        )
        result = validator.validate(test_case, "eitc", year=2023)
        assert result.success
        assert result.calculated_value is not None
        # Should have positive EITC
        assert result.calculated_value > 0

    def test_standard_deduction(self, validator):
        """Single filer should have standard deduction applied."""
        test_case = TestCase(
            name="Standard deduction",
            inputs={"earned_income": 50000, "filing_status": "SINGLE", "age": 35},
            expected={},
        )
        result = validator.validate(test_case, "standard_deduction", year=2023)
        assert result.success
        assert result.calculated_value is not None
        # 2023 standard deduction for single is $13,850
        assert result.calculated_value > 10000

    def test_joint_filer(self, validator):
        """Joint filers should have higher standard deduction."""
        test_case = TestCase(
            name="Joint filing",
            inputs={"earned_income": 100000, "filing_status": "JOINT", "age": 40},
            expected={},
        )
        result = validator.validate(test_case, "standard_deduction", year=2023)
        assert result.success
        assert result.calculated_value is not None
        # 2023 standard deduction for MFJ is $27,700
        assert result.calculated_value > 20000

    def test_unsupported_variable(self, validator):
        """Unsupported variable should return error."""
        test_case = TestCase(
            name="Test case",
            inputs={"earned_income": 50000},
            expected={},
        )
        result = validator.validate(test_case, "nonexistent_variable", year=2023)
        assert not result.success
        assert result.error is not None
        assert "not supported" in result.error.lower()

    def test_metadata_included(self, validator):
        """Result should include metadata about the calculation."""
        test_case = TestCase(
            name="Metadata test",
            inputs={"earned_income": 50000, "filing_status": "SINGLE"},
            expected={},
        )
        result = validator.validate(test_case, "agi", year=2023)
        assert result.success
        assert "tc_variable" in result.metadata
        assert "year" in result.metadata
        assert result.metadata["year"] == 2023

    def test_get_all_outputs(self, validator):
        """get_all_outputs should return multiple variables at once."""
        test_case = TestCase(
            name="All outputs",
            inputs={"earned_income": 30000, "filing_status": "SINGLE", "age": 30},
            expected={},
        )
        outputs = validator.get_all_outputs(test_case, year=2023)
        assert "eitc" in outputs
        assert "agi" in outputs
        assert "standard_deduction" in outputs
        assert "taxable_income" in outputs
        # AGI should equal wages
        assert outputs["agi"] is not None


class TestInputMapping:
    """Tests for input variable mapping."""

    @pytest.fixture
    def validator(self):
        return TaxCalculatorValidator()

    def test_age_mapping(self, validator):
        """Age should be properly mapped."""
        test_case = TestCase(
            name="Age test",
            inputs={"age": 45, "earned_income": 50000},
            expected={},
        )
        df = validator._build_input_dataframe(test_case, 2023)
        assert df["age_head"].iloc[0] == 45

    def test_wages_mapping(self, validator):
        """Wages should be mapped to e00200 variables."""
        test_case = TestCase(
            name="Wages test",
            inputs={"earned_income": 75000},
            expected={},
        )
        df = validator._build_input_dataframe(test_case, 2023)
        assert df["e00200"].iloc[0] == 75000
        assert df["e00200p"].iloc[0] == 75000

    def test_filing_status_single(self, validator):
        """Single filing status should map to MARS=1."""
        test_case = TestCase(
            name="Single status",
            inputs={"filing_status": "SINGLE"},
            expected={},
        )
        df = validator._build_input_dataframe(test_case, 2023)
        assert df["MARS"].iloc[0] == 1

    def test_filing_status_joint(self, validator):
        """Joint filing status should map to MARS=2."""
        test_case = TestCase(
            name="Joint status",
            inputs={"filing_status": "JOINT", "age": 40},
            expected={},
        )
        df = validator._build_input_dataframe(test_case, 2023)
        assert df["MARS"].iloc[0] == 2
        # Spouse age should be set when joint
        assert df["age_spouse"].iloc[0] == 40

    def test_children_mapping(self, validator):
        """Number of children should map to EIC and n24."""
        test_case = TestCase(
            name="Children test",
            inputs={"num_children": 2},
            expected={},
        )
        df = validator._build_input_dataframe(test_case, 2023)
        assert df["EIC"].iloc[0] == 2
        assert df["n24"].iloc[0] == 2

    def test_children_capped_at_3_for_eic(self, validator):
        """EIC should be capped at 3 even with more children."""
        test_case = TestCase(
            name="Many children",
            inputs={"num_children": 5},
            expected={},
        )
        df = validator._build_input_dataframe(test_case, 2023)
        assert df["EIC"].iloc[0] == 3  # Capped at 3
        assert df["n24"].iloc[0] == 5  # Actual count for CTC
