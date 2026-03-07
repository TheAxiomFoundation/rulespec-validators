"""Tests for Yale Tax-Simulator validator."""

from pathlib import Path
from unittest.mock import patch

import pytest

from cosilico_validators import TestCase, ValidatorType
from cosilico_validators.validators.yale import (
    SUPPORTED_VARIABLES,
    VARIABLE_MAPPING,
    YaleTaxValidator,
)


class TestYaleTaxValidatorConfig:
    """Test configuration and initialization."""

    def test_variable_mapping_includes_core_variables(self):
        """Ensure core tax variables are mapped."""
        assert "eitc" in VARIABLE_MAPPING
        assert "income_tax" in VARIABLE_MAPPING
        assert "ctc" in VARIABLE_MAPPING
        assert "agi" in VARIABLE_MAPPING
        assert "standard_deduction" in VARIABLE_MAPPING
        assert "taxable_income" in VARIABLE_MAPPING

    def test_supported_variables_complete(self):
        """Check that supported variables include both keys and values."""
        assert "eitc" in SUPPORTED_VARIABLES
        assert "earned_income_credit" in SUPPORTED_VARIABLES
        assert "federal_income_tax" in SUPPORTED_VARIABLES

    def test_validator_type_is_supplementary(self):
        """Yale validator should be supplementary (additional signal)."""
        # Can't instantiate without Tax-Simulator, check class attribute
        assert YaleTaxValidator.validator_type == ValidatorType.SUPPLEMENTARY

    def test_validator_name(self):
        """Check validator name."""
        assert YaleTaxValidator.name == "Yale Tax-Simulator"


class TestFilingStatusMapping:
    """Test filing status conversion."""

    @pytest.fixture
    def validator_with_mocks(self):
        """Create validator with mocked dependencies."""
        with (
            patch.object(YaleTaxValidator, "_resolve_path") as mock_path,
            patch.object(YaleTaxValidator, "_check_r_available"),
        ):
            mock_path.return_value = Path("/mock/Tax-Simulator")
            validator = YaleTaxValidator()
            return validator

    def test_single_filing_status(self, validator_with_mocks):
        """Test single filing status maps to MARS=1."""
        assert validator_with_mocks._map_filing_status("SINGLE") == 1

    def test_joint_filing_status(self, validator_with_mocks):
        """Test joint filing status maps to MARS=2."""
        assert validator_with_mocks._map_filing_status("JOINT") == 2
        assert validator_with_mocks._map_filing_status("MARRIED_FILING_JOINTLY") == 2

    def test_separate_filing_status(self, validator_with_mocks):
        """Test separate filing status maps to MARS=3."""
        assert validator_with_mocks._map_filing_status("MARRIED_FILING_SEPARATELY") == 3
        assert validator_with_mocks._map_filing_status("SEPARATE") == 3

    def test_head_of_household(self, validator_with_mocks):
        """Test head of household maps to MARS=4."""
        assert validator_with_mocks._map_filing_status("HEAD_OF_HOUSEHOLD") == 4

    def test_unknown_defaults_to_single(self, validator_with_mocks):
        """Test unknown filing status defaults to single."""
        assert validator_with_mocks._map_filing_status("UNKNOWN") == 1


class TestSupportsVariable:
    """Test variable support checking."""

    @pytest.fixture
    def validator_with_mocks(self):
        """Create validator with mocked dependencies."""
        with (
            patch.object(YaleTaxValidator, "_resolve_path") as mock_path,
            patch.object(YaleTaxValidator, "_check_r_available"),
        ):
            mock_path.return_value = Path("/mock/Tax-Simulator")
            validator = YaleTaxValidator()
            return validator

    def test_supports_eitc(self, validator_with_mocks):
        """Test EITC is supported."""
        assert validator_with_mocks.supports_variable("eitc")
        assert validator_with_mocks.supports_variable("EITC")
        assert validator_with_mocks.supports_variable("earned_income_credit")

    def test_supports_income_tax(self, validator_with_mocks):
        """Test income tax is supported."""
        assert validator_with_mocks.supports_variable("income_tax")
        assert validator_with_mocks.supports_variable("federal_income_tax")

    def test_supports_ctc(self, validator_with_mocks):
        """Test child tax credit is supported."""
        assert validator_with_mocks.supports_variable("ctc")
        assert validator_with_mocks.supports_variable("child_tax_credit")

    def test_supports_agi(self, validator_with_mocks):
        """Test AGI is supported."""
        assert validator_with_mocks.supports_variable("agi")
        assert validator_with_mocks.supports_variable("adjusted_gross_income")

    def test_supports_standard_deduction(self, validator_with_mocks):
        """Test standard deduction is supported."""
        assert validator_with_mocks.supports_variable("standard_deduction")

    def test_supports_taxable_income(self, validator_with_mocks):
        """Test taxable income is supported."""
        assert validator_with_mocks.supports_variable("taxable_income")

    def test_unsupported_variable(self, validator_with_mocks):
        """Test unsupported variable returns False."""
        assert not validator_with_mocks.supports_variable("state_sales_tax")
        assert not validator_with_mocks.supports_variable("foreign_tax_credit")


class TestValidateWithMocks:
    """Test validate method with mocked R execution."""

    @pytest.fixture
    def validator_with_mocks(self):
        """Create validator with mocked dependencies."""
        with (
            patch.object(YaleTaxValidator, "_resolve_path") as mock_path,
            patch.object(YaleTaxValidator, "_check_r_available"),
        ):
            mock_path.return_value = Path("/mock/Tax-Simulator")
            validator = YaleTaxValidator()
            return validator

    def test_unsupported_variable_returns_error(self, validator_with_mocks):
        """Test that unsupported variables return error result."""
        test_case = TestCase(
            name="Test",
            inputs={"earned_income": 30000},
            expected={"state_sales_tax": 1000},
        )
        result = validator_with_mocks.validate(test_case, "state_sales_tax", 2024)

        assert not result.success
        assert "not supported" in result.error.lower()

    @patch.object(YaleTaxValidator, "_run_simulator")
    def test_successful_validation(self, mock_run, validator_with_mocks):
        """Test successful validation with mocked simulator."""
        mock_run.return_value = {
            "eitc": 1502.0,
            "income_tax": 2500.0,
            "agi": 30000.0,
        }

        test_case = TestCase(
            name="EITC Test",
            inputs={"earned_income": 30000, "num_children": 2},
            expected={"eitc": 1502},
        )
        result = validator_with_mocks.validate(test_case, "eitc", 2024)

        assert result.success
        assert result.calculated_value == 1502.0
        assert result.validator_name == "Yale Tax-Simulator"

    @patch.object(YaleTaxValidator, "_run_simulator")
    def test_variable_not_in_output(self, mock_run, validator_with_mocks):
        """Test when requested variable is not in output."""
        mock_run.return_value = {
            "eitc": 1502.0,
            # No "amt" in output
        }

        test_case = TestCase(
            name="AMT Test",
            inputs={"earned_income": 500000},
            expected={"amt": 10000},
        )
        result = validator_with_mocks.validate(test_case, "amt", 2024)

        assert not result.success
        assert "not found" in result.error.lower()


class TestTaxUnitInputCreation:
    """Test creation of tax unit input files."""

    @pytest.fixture
    def validator_with_mocks(self):
        """Create validator with mocked dependencies."""
        with (
            patch.object(YaleTaxValidator, "_resolve_path") as mock_path,
            patch.object(YaleTaxValidator, "_check_r_available"),
        ):
            mock_path.return_value = Path("/mock/Tax-Simulator")
            validator = YaleTaxValidator()
            return validator

    def test_creates_csv_with_correct_fields(self, validator_with_mocks, tmp_path):
        """Test that input CSV has required fields."""
        import csv

        test_case = TestCase(
            name="Basic",
            inputs={
                "earned_income": 50000,
                "filing_status": "JOINT",
                "num_children": 2,
                "age": 35,
            },
            expected={"eitc": 0},
        )

        input_file = validator_with_mocks._create_tax_unit_input(test_case, 2024, tmp_path)

        with open(input_file) as f:
            reader = csv.DictReader(f)
            row = next(reader)

            assert row["RECID"] == "1"
            assert row["FLPDYR"] == "2024"
            assert row["MARS"] == "2"  # Joint
            assert row["e00200"] == "50000"
            assert row["n24"] == "2"
            assert row["age_head"] == "35"
