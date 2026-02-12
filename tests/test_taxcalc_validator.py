"""Tests for Tax-Calculator validator - full coverage."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from cosilico_validators.validators.base import TestCase
from cosilico_validators.validators.taxcalc import (
    TaxCalculatorValidator,
)


class TestGetTcModule:
    def test_import_error(self):
        v = TaxCalculatorValidator()
        with patch.dict("sys.modules", {"taxcalc": None}), \
             pytest.raises(ImportError, match="taxcalc not installed"):
            v._get_tc_module()

    def test_cached(self):
        v = TaxCalculatorValidator()
        v._tc_module = "cached"
        assert v._get_tc_module() == "cached"

    def test_import_success(self):
        v = TaxCalculatorValidator()
        mock_tc = MagicMock()
        with patch.dict("sys.modules", {"taxcalc": mock_tc}):
            result = v._get_tc_module()
            assert result is mock_tc


class TestBuildInputDataframe:
    def test_basic_wages(self):
        v = TaxCalculatorValidator()
        tc = TestCase(name="test", inputs={"earned_income": 50000}, expected={})
        df = v._build_input_dataframe(tc, 2024)
        assert df["e00200"].iloc[0] == 50000
        assert df["e00200p"].iloc[0] == 50000
        assert df["FLPDYR"].iloc[0] == 2024

    def test_spouse_wages(self):
        v = TaxCalculatorValidator()
        tc = TestCase(name="test", inputs={
            "earned_income": 50000,
            "spouse_wages": 30000,
            "filing_status": "JOINT",
        }, expected={})
        df = v._build_input_dataframe(tc, 2024)
        assert df["e00200s"].iloc[0] == 30000
        assert df["e00200"].iloc[0] == 80000

    def test_interest_income(self):
        v = TaxCalculatorValidator()
        tc = TestCase(name="test", inputs={"interest_income": 5000}, expected={})
        df = v._build_input_dataframe(tc, 2024)
        assert df["e00300"].iloc[0] == 5000

    def test_dividend_income(self):
        v = TaxCalculatorValidator()
        tc = TestCase(name="test", inputs={"dividend_income": 3000}, expected={})
        df = v._build_input_dataframe(tc, 2024)
        assert df["e00600"].iloc[0] == 3000
        assert df["e00650"].iloc[0] == 3000

    def test_social_security(self):
        v = TaxCalculatorValidator()
        tc = TestCase(name="test", inputs={"social_security": 15000}, expected={})
        df = v._build_input_dataframe(tc, 2024)
        assert df["e02400"].iloc[0] == 15000

    def test_self_employment(self):
        v = TaxCalculatorValidator()
        tc = TestCase(name="test", inputs={"self_employment": 40000}, expected={})
        df = v._build_input_dataframe(tc, 2024)
        assert df["e00900"].iloc[0] == 40000
        assert df["e00900p"].iloc[0] == 40000

    def test_pension_income(self):
        v = TaxCalculatorValidator()
        tc = TestCase(name="test", inputs={"pension_income": 20000}, expected={})
        df = v._build_input_dataframe(tc, 2024)
        assert df["e01500"].iloc[0] == 20000

    def test_salt_deduction(self):
        v = TaxCalculatorValidator()
        tc = TestCase(name="test", inputs={"salt": 10000}, expected={})
        df = v._build_input_dataframe(tc, 2024)
        assert df["e18400"].iloc[0] == 10000

    def test_property_tax(self):
        v = TaxCalculatorValidator()
        tc = TestCase(name="test", inputs={"property_tax": 5000}, expected={})
        df = v._build_input_dataframe(tc, 2024)
        assert df["e18500"].iloc[0] == 5000

    def test_mortgage_interest(self):
        v = TaxCalculatorValidator()
        tc = TestCase(name="test", inputs={"mortgage_interest": 12000}, expected={})
        df = v._build_input_dataframe(tc, 2024)
        assert df["e19200"].iloc[0] == 12000

    def test_charitable(self):
        v = TaxCalculatorValidator()
        tc = TestCase(name="test", inputs={"charitable": 2000}, expected={})
        df = v._build_input_dataframe(tc, 2024)
        assert df["e19800"].iloc[0] == 2000

    def test_n24_directly(self):
        v = TaxCalculatorValidator()
        tc = TestCase(name="test", inputs={"n24": 2}, expected={})
        df = v._build_input_dataframe(tc, 2024)
        assert df["n24"].iloc[0] == 2

    def test_spouse_age_with_joint(self):
        v = TaxCalculatorValidator()
        tc = TestCase(name="test", inputs={
            "filing_status": "JOINT", "age": 35
        }, expected={})
        df = v._build_input_dataframe(tc, 2024)
        assert df["age_spouse"].iloc[0] == 35

    def test_spouse_age_explicit(self):
        v = TaxCalculatorValidator()
        tc = TestCase(name="test", inputs={
            "filing_status": "JOINT", "age": 35, "spouse_age": 32
        }, expected={})
        df = v._build_input_dataframe(tc, 2024)
        assert df["age_spouse"].iloc[0] == 32


class TestValidateWithMocks:
    def test_validate_success(self):
        v = TaxCalculatorValidator()
        mock_tc = MagicMock()
        mock_recs = MagicMock()
        mock_tc.Records.return_value = mock_recs
        mock_pol = MagicMock()
        mock_tc.Policy.return_value = mock_pol
        mock_calc = MagicMock()
        mock_tc.Calculator.return_value = mock_calc
        mock_calc.dataframe.return_value = pd.DataFrame({"eitc": [500.0]})
        v._tc_module = mock_tc

        tc = TestCase(name="test", inputs={"earned_income": 30000}, expected={})
        result = v.validate(tc, "eitc", 2024)
        assert result.success
        assert result.calculated_value == 500.0

    def test_validate_unsupported_variable(self):
        v = TaxCalculatorValidator()
        v._tc_module = MagicMock()  # Mock the module so it doesn't fail on import
        tc = TestCase(name="test", inputs={}, expected={})
        result = v.validate(tc, "nonexistent_xyz_variable", 2024)
        assert not result.success
        assert "not supported" in result.error

    def test_validate_exception(self):
        v = TaxCalculatorValidator()
        mock_tc = MagicMock()
        mock_tc.Records.side_effect = Exception("Records failed")
        v._tc_module = mock_tc

        tc = TestCase(name="test", inputs={"earned_income": 30000}, expected={})
        result = v.validate(tc, "eitc", 2024)
        assert not result.success
        assert "Tax-Calculator execution failed" in result.error

    def test_validate_metadata(self):
        v = TaxCalculatorValidator()
        mock_tc = MagicMock()
        mock_calc = MagicMock()
        mock_tc.Calculator.return_value = mock_calc
        mock_calc.dataframe.return_value = pd.DataFrame({"eitc": [500.0]})
        v._tc_module = mock_tc

        tc = TestCase(name="test", inputs={"earned_income": 30000, "filing_status": "SINGLE"}, expected={})
        result = v.validate(tc, "eitc", 2024)
        assert result.metadata["tc_variable"] == "eitc"
        assert result.metadata["year"] == 2024


class TestGetAllOutputs:
    def test_get_all_outputs_success(self):
        v = TaxCalculatorValidator()
        mock_tc = MagicMock()
        mock_calc = MagicMock()
        mock_tc.Calculator.return_value = mock_calc
        result_df = pd.DataFrame({
            "eitc": [500.0],
            "iitax": [3000.0],
            "c00100": [50000.0],
            "standard": [14600.0],
            "c04800": [35400.0],
        })
        mock_calc.dataframe.return_value = result_df
        v._tc_module = mock_tc

        tc = TestCase(name="test", inputs={"earned_income": 50000}, expected={})
        outputs = v.get_all_outputs(tc, 2024)
        assert "eitc" in outputs
        assert "agi" in outputs
        assert outputs["eitc"] == 500.0

    def test_get_all_outputs_error(self):
        v = TaxCalculatorValidator()
        mock_tc = MagicMock()
        mock_tc.Records.side_effect = Exception("Records failed")
        v._tc_module = mock_tc

        tc = TestCase(name="test", inputs={}, expected={})
        outputs = v.get_all_outputs(tc, 2024)
        # All values should be None on error
        for val in outputs.values():
            assert val is None
