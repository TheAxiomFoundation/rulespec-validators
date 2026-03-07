"""PolicyEngine validator - uses policyengine-us package."""

from typing import Any

from cosilico_validators.validators.base import (
    BaseValidator,
    TestCase,
    ValidatorResult,
    ValidatorType,
)

# Variable mapping from common names to PolicyEngine variable names
VARIABLE_MAPPING = {
    "eitc": "eitc",
    "earned_income_credit": "eitc",
    "ctc": "ctc",
    "child_tax_credit": "ctc",
    "income_tax": "income_tax",
    "federal_income_tax": "income_tax",
    "state_income_tax": "state_income_tax",
    "snap": "snap",
    "snap_benefits": "snap",
    "medicaid": "medicaid",
    "tanf": "tanf",
    # Standard deduction
    "standard_deduction": "standard_deduction",
    "basic_standard_deduction": "basic_standard_deduction",
    "additional_standard_deduction": "additional_standard_deduction",
    # Adjusted gross income
    "agi": "adjusted_gross_income",
    "adjusted_gross_income": "adjusted_gross_income",
}

# Supported variables
SUPPORTED_VARIABLES = set(VARIABLE_MAPPING.keys()) | set(VARIABLE_MAPPING.values())


class PolicyEngineValidator(BaseValidator):
    """Validator using PolicyEngine US microsimulation."""

    name = "PolicyEngine"
    validator_type = ValidatorType.REFERENCE
    supported_variables = SUPPORTED_VARIABLES

    def __init__(self):
        self._simulation_class = None

    def _get_simulation_class(self):
        """Lazy load Simulation to avoid import overhead."""
        if self._simulation_class is None:
            try:
                from policyengine_us import Simulation

                self._simulation_class = Simulation
            except ImportError as e:
                raise ImportError(
                    "policyengine-us not installed. Install with: pip install cosilico-validators[policyengine]"
                ) from e
        return self._simulation_class

    def supports_variable(self, variable: str) -> bool:
        return variable.lower() in SUPPORTED_VARIABLES

    def _build_situation(self, test_case: TestCase, year: int) -> dict[str, Any]:
        """Convert test case inputs to PolicyEngine situation format."""
        inputs = test_case.inputs
        year_str = str(year)

        # Basic household structure
        situation: dict[str, Any] = {
            "people": {"adult": {"age": {year_str: 30}}},
            "tax_units": {"tax_unit": {"members": ["adult"]}},
            "spm_units": {"spm_unit": {"members": ["adult"]}},
            "households": {"household": {"members": ["adult"], "state_name": {year_str: "CA"}}},
            "families": {"family": {"members": ["adult"]}},
            "marital_units": {"marital_unit": {"members": ["adult"]}},
        }

        # Map common input names to PE variables
        input_handlers = {
            "age": lambda v: self._set_person_var(situation, "age", v, year_str),
            "age_at_end_of_year": lambda v: self._set_person_var(situation, "age", v, year_str),
            "earned_income": lambda v: self._set_person_var(situation, "employment_income", v, year_str),
            "employment_income": lambda v: self._set_person_var(situation, "employment_income", v, year_str),
            "wages": lambda v: self._set_person_var(situation, "employment_income", v, year_str),
            "filing_status": lambda v: self._handle_filing_status(situation, v, year_str),
            "eitc_qualifying_children_count": lambda v: self._add_children(situation, v, year_str),
            "num_children": lambda v: self._add_children(situation, v, year_str),
            "state": lambda v: self._set_state(situation, v, year_str),
            "state_name": lambda v: self._set_state(situation, v, year_str),
        }

        for key, value in inputs.items():
            key_lower = key.lower()
            if key_lower in input_handlers:
                input_handlers[key_lower](value)

        return situation

    def _set_person_var(self, situation: dict, var: str, value: Any, year_str: str) -> None:
        """Set a person-level variable."""
        situation["people"]["adult"][var] = {year_str: value}

    def _handle_filing_status(self, situation: dict, status: str, year_str: str) -> None:
        """Handle filing status - add spouse if joint."""
        if status.upper() in ["JOINT", "MARRIED_FILING_JOINTLY"]:
            situation["people"]["spouse"] = {"age": {year_str: 30}}
            for entity in [
                "tax_units",
                "spm_units",
                "households",
                "families",
                "marital_units",
            ]:
                entity_name = list(situation[entity].keys())[0]
                situation[entity][entity_name]["members"].append("spouse")

    def _add_children(self, situation: dict, count: int, year_str: str) -> None:
        """Add qualifying children to the household."""
        for i in range(count):
            child_id = f"child_{i}"
            situation["people"][child_id] = {
                "age": {year_str: 5},
                "is_tax_unit_dependent": {year_str: True},
            }
            for entity in ["tax_units", "spm_units", "households", "families"]:
                entity_name = list(situation[entity].keys())[0]
                situation[entity][entity_name]["members"].append(child_id)

    def _set_state(self, situation: dict, state: str, year_str: str) -> None:
        """Set the household state."""
        situation["households"]["household"]["state_name"] = {year_str: state}

    def validate(self, test_case: TestCase, variable: str, year: int = 2024) -> ValidatorResult:
        """Run validation using PolicyEngine."""
        Simulation = self._get_simulation_class()

        # Map variable name
        pe_variable = VARIABLE_MAPPING.get(variable.lower(), variable.lower())

        try:
            situation = self._build_situation(test_case, year)
            sim = Simulation(situation=situation)
            value = sim.calculate(pe_variable, year)

            # Handle array output
            calculated = float(value[0]) if hasattr(value, "__len__") and len(value) > 0 else float(value)

            return ValidatorResult(
                validator_name=self.name,
                validator_type=self.validator_type,
                calculated_value=calculated,
                metadata={"pe_variable": pe_variable, "year": year},
            )

        except Exception as e:
            return ValidatorResult(
                validator_name=self.name,
                validator_type=self.validator_type,
                calculated_value=None,
                error=str(e),
                metadata={"pe_variable": pe_variable, "year": year},
            )
