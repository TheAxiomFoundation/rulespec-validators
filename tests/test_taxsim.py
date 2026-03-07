"""Tests for TAXSIM validator."""

import pytest

from cosilico_validators.validators.base import TestCase, ValidatorType
from cosilico_validators.validators.taxsim import (
    MSTAT_CODES,
    STATE_CODES,
    TAXSIM_COLUMNS,
    TAXSIM_OUTPUT_VARS,
    TaxsimValidator,
)


class TestTaxsimOutputVars:
    """Test TAXSIM output variable mappings."""

    def test_eitc_mapping(self):
        assert TAXSIM_OUTPUT_VARS["eitc"] == "v25"
        assert TAXSIM_OUTPUT_VARS["earned_income_credit"] == "v25"

    def test_ctc_mapping(self):
        assert TAXSIM_OUTPUT_VARS["ctc"] == "v22"
        assert TAXSIM_OUTPUT_VARS["child_tax_credit"] == "v22"
        assert TAXSIM_OUTPUT_VARS["actc"] == "v23"
        assert TAXSIM_OUTPUT_VARS["refundable_ctc"] == "v23"

    def test_income_mappings(self):
        assert TAXSIM_OUTPUT_VARS["agi"] == "v10"
        assert TAXSIM_OUTPUT_VARS["adjusted_gross_income"] == "v10"
        assert TAXSIM_OUTPUT_VARS["taxable_income"] == "v18"

    def test_tax_mappings(self):
        assert TAXSIM_OUTPUT_VARS["federal_income_tax"] == "fiitax"
        assert TAXSIM_OUTPUT_VARS["income_tax"] == "fiitax"
        assert TAXSIM_OUTPUT_VARS["state_income_tax"] == "siitax"

    def test_standard_deduction_mapping(self):
        assert TAXSIM_OUTPUT_VARS["standard_deduction"] == "v13"
        assert TAXSIM_OUTPUT_VARS["zero_bracket_amount"] == "v13"


class TestStateCodes:
    """Test state FIPS code mappings."""

    def test_common_states(self):
        assert STATE_CODES["CA"] == 6
        assert STATE_CODES["NY"] == 36
        assert STATE_CODES["TX"] == 48
        assert STATE_CODES["FL"] == 12

    def test_dc_code(self):
        assert STATE_CODES["DC"] == 11


class TestMstatCodes:
    """Test filing status code mappings."""

    def test_single(self):
        assert MSTAT_CODES["SINGLE"] == 1

    def test_joint(self):
        assert MSTAT_CODES["JOINT"] == 2
        assert MSTAT_CODES["MARRIED_FILING_JOINTLY"] == 2

    def test_head_of_household(self):
        assert MSTAT_CODES["HEAD_OF_HOUSEHOLD"] == 3
        assert MSTAT_CODES["HOH"] == 3

    def test_separate(self):
        assert MSTAT_CODES["MARRIED_FILING_SEPARATELY"] == 6
        assert MSTAT_CODES["SEPARATE"] == 6
        assert MSTAT_CODES["MFS"] == 6


class TestTaxsimValidatorInit:
    """Test TaxsimValidator initialization."""

    def test_default_mode_is_web(self):
        validator = TaxsimValidator()
        assert validator.mode == "web"
        assert validator.taxsim_path is None

    def test_web_mode_no_path_needed(self):
        validator = TaxsimValidator(mode="web")
        assert validator.mode == "web"
        assert validator.taxsim_path is None

    def test_default_retries_and_timeout(self):
        validator = TaxsimValidator()
        assert validator.max_retries == 3
        assert validator.timeout == 60

    def test_custom_retries_and_timeout(self):
        validator = TaxsimValidator(max_retries=5, timeout=120)
        assert validator.max_retries == 5
        assert validator.timeout == 120

    def test_validator_properties(self):
        validator = TaxsimValidator()
        assert validator.name == "TAXSIM"
        assert validator.validator_type == ValidatorType.REFERENCE
        assert len(validator.supported_variables) > 0


class TestTaxsimValidatorSupportsVariable:
    """Test variable support checking."""

    def test_supports_eitc(self):
        validator = TaxsimValidator()
        assert validator.supports_variable("eitc")
        assert validator.supports_variable("EITC")
        assert validator.supports_variable("Eitc")

    def test_supports_ctc(self):
        validator = TaxsimValidator()
        assert validator.supports_variable("ctc")
        assert validator.supports_variable("child_tax_credit")

    def test_supports_income_variables(self):
        validator = TaxsimValidator()
        assert validator.supports_variable("agi")
        assert validator.supports_variable("taxable_income")
        assert validator.supports_variable("income_tax")

    def test_supports_standard_deduction(self):
        validator = TaxsimValidator()
        assert validator.supports_variable("standard_deduction")

    def test_does_not_support_unknown(self):
        validator = TaxsimValidator()
        assert not validator.supports_variable("unknown_variable")
        assert not validator.supports_variable("snap")  # Not a TAXSIM variable


class TestTaxsimValidatorBuildInput:
    """Test input building for TAXSIM."""

    def test_basic_input_building(self):
        validator = TaxsimValidator()
        test_case = TestCase(
            name="Basic test",
            inputs={"earned_income": 30000},
            expected={"eitc": 500},
        )
        result = validator._build_taxsim_input(test_case, 2023)

        assert result["year"] == 2023
        assert result["pwages"] == 30000
        assert result["idtl"] == 2  # Full output

    def test_filing_status_single(self):
        validator = TaxsimValidator()
        test_case = TestCase(
            name="Single filer",
            inputs={"earned_income": 30000, "filing_status": "SINGLE"},
            expected={},
        )
        result = validator._build_taxsim_input(test_case, 2023)
        assert result["mstat"] == 1

    def test_filing_status_joint(self):
        validator = TaxsimValidator()
        test_case = TestCase(
            name="Joint filer",
            inputs={"earned_income": 50000, "filing_status": "JOINT"},
            expected={},
        )
        result = validator._build_taxsim_input(test_case, 2023)
        assert result["mstat"] == 2
        assert result["sage"] > 0  # Spouse age should be set

    def test_filing_status_hoh(self):
        validator = TaxsimValidator()
        test_case = TestCase(
            name="Head of household",
            inputs={"earned_income": 30000, "filing_status": "HEAD_OF_HOUSEHOLD"},
            expected={},
        )
        result = validator._build_taxsim_input(test_case, 2023)
        assert result["mstat"] == 3

    def test_children_creates_ages(self):
        validator = TaxsimValidator()
        test_case = TestCase(
            name="With children",
            inputs={"earned_income": 30000, "num_children": 2},
            expected={},
        )
        result = validator._build_taxsim_input(test_case, 2023)
        assert result["depx"] == 2
        assert result["age1"] == 10
        assert result["age2"] == 10

    def test_state_code_from_abbreviation(self):
        validator = TaxsimValidator()
        test_case = TestCase(
            name="California",
            inputs={"earned_income": 30000, "state": "CA"},
            expected={},
        )
        result = validator._build_taxsim_input(test_case, 2023)
        assert result["state"] == 6

    def test_state_code_from_name(self):
        validator = TaxsimValidator()
        test_case = TestCase(
            name="New York",
            inputs={"earned_income": 30000, "state_name": "NY"},
            expected={},
        )
        result = validator._build_taxsim_input(test_case, 2023)
        assert result["state"] == 36


class TestTaxsimValidatorCSV:
    """Test CSV generation for TAXSIM."""

    def test_csv_string_creation(self):
        validator = TaxsimValidator()
        taxsim_input = {
            "taxsimid": 1,
            "year": 2023,
            "mstat": 1,
            "pwages": 30000,
            "idtl": 2,
        }
        csv_str = validator._create_csv_string(taxsim_input)

        # Should have header row
        lines = csv_str.strip().split("\n")
        assert len(lines) == 2

        # Header should have all columns
        header = lines[0]
        assert "taxsimid" in header
        assert "year" in header
        assert "pwages" in header

    def test_csv_columns_order(self):
        # Verify column order matches TAXSIM expectations
        assert TAXSIM_COLUMNS[0] == "taxsimid"
        assert TAXSIM_COLUMNS[1] == "year"
        assert TAXSIM_COLUMNS[2] == "state"
        assert TAXSIM_COLUMNS[3] == "mstat"
        assert "idtl" in TAXSIM_COLUMNS  # Full output flag


class TestTaxsimValidatorParseOutput:
    """Test output parsing from TAXSIM."""

    def test_parse_fiitax(self):
        validator = TaxsimValidator()
        output = "taxsimid,year,fiitax,v25\n1,2023,5000.00,600.00"
        result = validator._parse_output(output, "federal_income_tax")
        assert result == 5000.00

    def test_parse_eitc(self):
        validator = TaxsimValidator()
        output = "taxsimid,year,fiitax,v25\n1,2023,5000.00,600.00"
        result = validator._parse_output(output, "eitc")
        assert result == 600.00

    def test_parse_missing_variable(self):
        validator = TaxsimValidator()
        output = "taxsimid,year,fiitax\n1,2023,5000.00"
        result = validator._parse_output(output, "eitc")
        assert result is None


class TestTaxsimValidatorValidate:
    """Test validation execution."""

    def test_year_validation_too_old(self):
        validator = TaxsimValidator()
        test_case = TestCase(
            name="Test",
            inputs={"earned_income": 30000},
            expected={},
        )
        result = validator.validate(test_case, "eitc", year=1959)
        assert result.error is not None
        assert "1960-2023" in result.error

    def test_year_validation_too_new(self):
        validator = TaxsimValidator()
        test_case = TestCase(
            name="Test",
            inputs={"earned_income": 30000},
            expected={},
        )
        result = validator.validate(test_case, "eitc", year=2024)
        assert result.error is not None
        assert "1960-2023" in result.error

    def test_unsupported_variable(self):
        validator = TaxsimValidator()
        test_case = TestCase(
            name="Test",
            inputs={"earned_income": 30000},
            expected={},
        )
        result = validator.validate(test_case, "snap", year=2023)
        assert result.error is not None
        assert "not supported" in result.error


class TestTaxsimValidatorIntegration:
    """Integration tests for TAXSIM validator (require network access)."""

    @pytest.mark.integration
    def test_web_api_eitc(self):
        """Test EITC calculation via web API.

        This test requires network access to taxsim.nber.org.
        Run with: pytest -m integration
        """
        validator = TaxsimValidator(mode="web")
        test_case = TestCase(
            name="EITC single filer",
            inputs={
                "earned_income": 15000,
                "filing_status": "SINGLE",
                "eitc_qualifying_children_count": 0,
            },
            expected={"eitc": 600},  # Approximate
        )
        result = validator.validate(test_case, "eitc", year=2023)

        assert result.success, f"Failed: {result.error}"
        assert result.calculated_value is not None
        assert result.calculated_value >= 0  # EITC should be non-negative
        assert result.metadata.get("mode") == "web"

    @pytest.mark.integration
    def test_web_api_batch(self):
        """Test batch validation via web API.

        This test requires network access to taxsim.nber.org.
        Run with: pytest -m integration
        """
        validator = TaxsimValidator(mode="web")
        test_cases = [
            TestCase(
                name=f"Case {i}",
                inputs={"earned_income": income, "filing_status": "SINGLE"},
                expected={},
            )
            for i, income in enumerate([15000, 25000, 40000])
        ]
        results = validator.batch_validate(test_cases, "eitc", year=2023)

        assert len(results) == 3
        for result in results:
            assert result.success, f"Failed: {result.error}"
            assert result.calculated_value is not None
