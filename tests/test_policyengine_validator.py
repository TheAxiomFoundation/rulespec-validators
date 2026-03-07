"""Tests for PolicyEngine validator - full coverage."""

from unittest.mock import MagicMock, patch

import pytest

from cosilico_validators.validators.base import TestCase, ValidatorType
from cosilico_validators.validators.policyengine import (
    PolicyEngineValidator,
)


class TestPolicyEngineValidatorInit:
    def test_init(self):
        v = PolicyEngineValidator()
        assert v._simulation_class is None
        assert v.name == "PolicyEngine"
        assert v.validator_type == ValidatorType.REFERENCE

    def test_get_simulation_class_import_error(self):
        v = PolicyEngineValidator()
        with (
            patch.dict("sys.modules", {"policyengine_us": None}),
            pytest.raises(ImportError, match="policyengine-us not installed"),
        ):
            v._get_simulation_class()

    def test_get_simulation_class_success(self):
        v = PolicyEngineValidator()
        mock_module = MagicMock()
        mock_module.Simulation = MagicMock()
        with patch.dict("sys.modules", {"policyengine_us": mock_module}):
            cls = v._get_simulation_class()
            assert cls is mock_module.Simulation

    def test_get_simulation_class_cached(self):
        v = PolicyEngineValidator()
        v._simulation_class = "cached"
        assert v._get_simulation_class() == "cached"


class TestSupportsVariable:
    def test_supports_eitc(self):
        v = PolicyEngineValidator()
        assert v.supports_variable("eitc")
        assert v.supports_variable("earned_income_credit")

    def test_supports_income_tax(self):
        v = PolicyEngineValidator()
        assert v.supports_variable("income_tax")
        assert v.supports_variable("federal_income_tax")

    def test_supports_standard_deduction(self):
        v = PolicyEngineValidator()
        assert v.supports_variable("standard_deduction")
        assert v.supports_variable("basic_standard_deduction")

    def test_unsupported(self):
        v = PolicyEngineValidator()
        assert not v.supports_variable("state_sales_tax")
        assert not v.supports_variable("foreign_tax_credit_xyz")


class TestBuildSituation:
    def _make_validator(self):
        return PolicyEngineValidator()

    def test_basic_situation(self):
        v = self._make_validator()
        tc = TestCase(name="test", inputs={"earned_income": 30000}, expected={})
        situation = v._build_situation(tc, 2024)
        assert "people" in situation
        assert "adult" in situation["people"]
        assert situation["people"]["adult"]["employment_income"] == {"2024": 30000}

    def test_filing_status_joint(self):
        v = self._make_validator()
        tc = TestCase(name="test", inputs={"filing_status": "JOINT"}, expected={})
        situation = v._build_situation(tc, 2024)
        assert "spouse" in situation["people"]
        # spouse in all entities
        for entity in ["tax_units", "spm_units", "households", "families", "marital_units"]:
            entity_name = list(situation[entity].keys())[0]
            assert "spouse" in situation[entity][entity_name]["members"]

    def test_filing_status_married_filing_jointly(self):
        v = self._make_validator()
        tc = TestCase(name="test", inputs={"filing_status": "MARRIED_FILING_JOINTLY"}, expected={})
        situation = v._build_situation(tc, 2024)
        assert "spouse" in situation["people"]

    def test_filing_status_single(self):
        v = self._make_validator()
        tc = TestCase(name="test", inputs={"filing_status": "SINGLE"}, expected={})
        situation = v._build_situation(tc, 2024)
        assert "spouse" not in situation["people"]

    def test_add_children(self):
        v = self._make_validator()
        tc = TestCase(name="test", inputs={"num_children": 3}, expected={})
        situation = v._build_situation(tc, 2024)
        assert "child_0" in situation["people"]
        assert "child_1" in situation["people"]
        assert "child_2" in situation["people"]
        assert situation["people"]["child_0"]["age"] == {"2024": 5}
        assert situation["people"]["child_0"]["is_tax_unit_dependent"] == {"2024": True}

    def test_set_state(self):
        v = self._make_validator()
        tc = TestCase(name="test", inputs={"state": "NY"}, expected={})
        situation = v._build_situation(tc, 2024)
        assert situation["households"]["household"]["state_name"] == {"2024": "NY"}

    def test_set_age(self):
        v = self._make_validator()
        tc = TestCase(name="test", inputs={"age": 45}, expected={})
        situation = v._build_situation(tc, 2024)
        assert situation["people"]["adult"]["age"] == {"2024": 45}


class TestValidate:
    def test_successful_validation(self):
        v = PolicyEngineValidator()
        mock_sim_cls = MagicMock()
        mock_sim_instance = MagicMock()
        mock_sim_cls.return_value = mock_sim_instance
        mock_sim_instance.calculate.return_value = [500.0]
        v._simulation_class = mock_sim_cls

        tc = TestCase(name="test", inputs={"earned_income": 30000}, expected={"eitc": 500})
        result = v.validate(tc, "eitc", 2024)
        assert result.success
        assert result.calculated_value == 500.0
        assert result.validator_name == "PolicyEngine"

    def test_validation_error(self):
        v = PolicyEngineValidator()
        mock_sim_cls = MagicMock()
        mock_sim_cls.side_effect = Exception("simulation failed")
        v._simulation_class = mock_sim_cls

        tc = TestCase(name="test", inputs={"earned_income": 30000}, expected={})
        result = v.validate(tc, "eitc", 2024)
        assert not result.success
        assert "simulation failed" in result.error

    def test_variable_mapping(self):
        v = PolicyEngineValidator()
        mock_sim_cls = MagicMock()
        mock_sim_instance = MagicMock()
        mock_sim_cls.return_value = mock_sim_instance
        mock_sim_instance.calculate.return_value = [100.0]
        v._simulation_class = mock_sim_cls

        tc = TestCase(name="test", inputs={"earned_income": 30000}, expected={})
        v.validate(tc, "earned_income_credit", 2024)
        mock_sim_instance.calculate.assert_called_with("eitc", 2024)

    def test_result_metadata(self):
        v = PolicyEngineValidator()
        mock_sim_cls = MagicMock()
        mock_sim_instance = MagicMock()
        mock_sim_cls.return_value = mock_sim_instance
        mock_sim_instance.calculate.return_value = [100.0]
        v._simulation_class = mock_sim_cls

        tc = TestCase(name="test", inputs={}, expected={})
        result = v.validate(tc, "eitc", 2024)
        assert result.metadata["pe_variable"] == "eitc"
        assert result.metadata["year"] == 2024

    def test_scalar_output(self):
        v = PolicyEngineValidator()
        mock_sim_cls = MagicMock()
        mock_sim_instance = MagicMock()
        mock_sim_cls.return_value = mock_sim_instance
        mock_sim_instance.calculate.return_value = 500.0
        v._simulation_class = mock_sim_cls

        tc = TestCase(name="test", inputs={}, expected={})
        result = v.validate(tc, "eitc", 2024)
        assert result.calculated_value == 500.0
