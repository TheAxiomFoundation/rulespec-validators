"""PSL Tax-Calculator validator - uses taxcalc package.

Tax-Calculator is an open-source microsimulation model for USA federal
income and payroll taxes from the Policy Simulation Library (PSL).

See: https://github.com/PSLmodels/Tax-Calculator
Docs: https://taxcalc.pslmodels.org/
"""

from typing import Any

import pandas as pd

from cosilico_validators.validators.base import (
    BaseValidator,
    TestCase,
    ValidatorResult,
    ValidatorType,
)

# Variable mapping from common names to Tax-Calculator variable names
# Output variables: https://taxcalc.pslmodels.org/guide/output_vars.html
VARIABLE_MAPPING = {
    # EITC
    "eitc": "eitc",
    "earned_income_credit": "eitc",
    # Income tax
    "income_tax": "iitax",
    "federal_income_tax": "iitax",
    "iitax": "iitax",
    # Child Tax Credit
    "ctc": "c07220",
    "child_tax_credit": "c07220",
    "ctc_refundable": "c11070",  # Refundable portion (ACTC)
    # Standard deduction
    "standard_deduction": "standard",
    "standard": "standard",
    # AGI
    "agi": "c00100",
    "adjusted_gross_income": "c00100",
    "c00100": "c00100",
    # Taxable income
    "taxable_income": "c04800",
    "c04800": "c04800",
    # Payroll tax
    "payroll_tax": "payrolltax",
    "payrolltax": "payrolltax",
    # AMT
    "amt": "c09600",
    "alternative_minimum_tax": "c09600",
    # Pre-credit tax
    "income_tax_before_credits": "c05800",
    "c05800": "c05800",
}

# Supported variables
SUPPORTED_VARIABLES = set(VARIABLE_MAPPING.keys()) | set(VARIABLE_MAPPING.values())

# Filing status mapping (MARS values)
# 1=single, 2=joint, 3=separate, 4=household-head, 5=widow(er)
MARS_MAPPING = {
    "SINGLE": 1,
    "JOINT": 2,
    "MARRIED_FILING_JOINTLY": 2,
    "MARRIED_FILING_SEPARATELY": 3,
    "SEPARATE": 3,
    "HEAD_OF_HOUSEHOLD": 4,
    "HOUSEHOLD_HEAD": 4,
    "WIDOW": 5,
    "WIDOWER": 5,
    "QUALIFYING_WIDOW": 5,
}


class TaxCalculatorValidator(BaseValidator):
    """Validator using PSL Tax-Calculator microsimulation.

    Tax-Calculator is a well-documented, open-source model maintained
    by the Policy Simulation Library. It supports analysis of federal
    income and payroll taxes for any tax year from 2013 onwards.

    This validator creates single-record calculations using custom
    input data, similar to how Tax-Cruncher operates.
    """

    name = "Tax-Calculator"
    validator_type = ValidatorType.SUPPLEMENTARY
    supported_variables = SUPPORTED_VARIABLES

    def __init__(self):
        self._tc_module = None

    def _get_tc_module(self):
        """Lazy load taxcalc to avoid import overhead."""
        if self._tc_module is None:
            try:
                import taxcalc as tc

                self._tc_module = tc
            except ImportError as e:
                raise ImportError("taxcalc not installed. Install with: pip install cosilico-validators[psl]") from e
        return self._tc_module

    def supports_variable(self, variable: str) -> bool:
        return variable.lower() in SUPPORTED_VARIABLES

    def _build_input_dataframe(self, test_case: TestCase, year: int) -> pd.DataFrame:
        """Convert test case inputs to Tax-Calculator input DataFrame.

        Tax-Calculator requires specific variable names. We create a
        single-record DataFrame with the necessary inputs.
        """
        inputs = test_case.inputs

        # Start with default values for a single filer
        tc_inputs: dict[str, Any] = {
            "RECID": 1,  # Record ID
            "FLPDYR": year,  # Tax year
            "MARS": 1,  # Filing status (single)
            "age_head": 30,  # Age of primary taxpayer
            "age_spouse": 0,  # Age of spouse (0 if none)
            # Wages - must include split between taxpayer/spouse
            "e00200": 0.0,  # Total wages
            "e00200p": 0.0,  # Taxpayer wages
            "e00200s": 0.0,  # Spouse wages
            # Business income
            "e00900": 0.0,
            "e00900p": 0.0,
            "e00900s": 0.0,
            # Farm income
            "e02100": 0.0,
            "e02100p": 0.0,
            "e02100s": 0.0,
            # Dividends
            "e00600": 0.0,  # Ordinary dividends
            "e00650": 0.0,  # Qualified dividends
            # Pensions
            "e01500": 0.0,  # Taxable pensions
            "e01700": 0.0,  # Pension distributions
            # Interest
            "e00300": 0.0,  # Taxable interest
            "e00400": 0.0,  # Tax-exempt interest
            # Other income
            "e01400": 0.0,  # Taxable IRA distributions
            "e02000": 0.0,  # Schedule E income
            "e02300": 0.0,  # Unemployment compensation
            "e02400": 0.0,  # Social security benefits
            # EITC variables
            "EIC": 0,  # Number of EITC qualifying children (0-3)
            "n24": 0,  # Number of CTC qualifying children
            "nu18": 0,  # Number of people under 18
            "nu13": 0,  # Number of dependents under 13
            "nu06": 0,  # Number of dependents under 6
            "elderly_dependents": 0,  # Number of elderly dependents
            # Child care expenses
            "f2441": 0,  # Number of children eligible for CDCTC
            "e32800": 0.0,  # Child care expenses
            # Deductions
            "e18400": 0.0,  # State and local income taxes
            "e18500": 0.0,  # Real estate taxes
            "e19200": 0.0,  # Mortgage interest
            "e19800": 0.0,  # Charitable contributions
            "e20100": 0.0,  # Miscellaneous deductions
            # Pass-through income
            "PT_SSTB_income": 0,  # SSTB indicator
            "e26270": 0.0,  # Partnership/S-corp income
            "k1bx14s": 0.0,  # Spouse K-1 income
            # Weight (set to 1 for single record)
            "s006": 1.0,
        }

        # Map common input names to Tax-Calculator variables
        for key, value in inputs.items():
            key_lower = key.lower()

            # Handle age
            if key_lower in ["age", "age_at_end_of_year", "age_head"]:
                tc_inputs["age_head"] = int(value)

            # Handle spouse age
            elif key_lower in ["spouse_age", "age_spouse"]:
                tc_inputs["age_spouse"] = int(value)

            # Handle wages/earned income
            elif key_lower in ["earned_income", "employment_income", "wages", "e00200"]:
                tc_inputs["e00200"] = float(value)
                tc_inputs["e00200p"] = float(value)  # Assign all to primary

            # Handle spouse wages
            elif key_lower in ["spouse_wages", "spouse_income", "e00200s"]:
                tc_inputs["e00200s"] = float(value)
                # Update total wages
                tc_inputs["e00200"] = tc_inputs["e00200p"] + float(value)

            # Handle filing status
            elif key_lower == "filing_status":
                status = str(value).upper().replace(" ", "_")
                tc_inputs["MARS"] = MARS_MAPPING.get(status, 1)

            # Handle number of qualifying children for EITC
            elif key_lower in [
                "eitc_qualifying_children_count",
                "num_children",
                "qualifying_children",
                "children",
                "eic",
            ]:
                num = int(value)
                tc_inputs["EIC"] = min(num, 3)  # EITC caps at 3
                tc_inputs["n24"] = num  # CTC uses actual count
                tc_inputs["nu18"] = num
                tc_inputs["nu13"] = num

            # Handle child ages for CTC (children must be under 17)
            elif key_lower in ["n24", "ctc_children"]:
                tc_inputs["n24"] = int(value)

            # Handle interest income
            elif key_lower in ["interest_income", "interest", "e00300"]:
                tc_inputs["e00300"] = float(value)

            # Handle dividend income
            elif key_lower in ["dividends", "dividend_income", "e00600"]:
                tc_inputs["e00600"] = float(value)
                tc_inputs["e00650"] = float(value)  # Assume qualified

            # Handle social security
            elif key_lower in ["social_security", "ss_income", "e02400"]:
                tc_inputs["e02400"] = float(value)

            # Handle business income
            elif key_lower in ["self_employment", "business_income", "e00900"]:
                tc_inputs["e00900"] = float(value)
                tc_inputs["e00900p"] = float(value)

            # Handle pension income
            elif key_lower in ["pension_income", "pension", "e01500"]:
                tc_inputs["e01500"] = float(value)

            # Handle state/local taxes (SALT)
            elif key_lower in ["salt", "state_local_taxes", "e18400"]:
                tc_inputs["e18400"] = float(value)

            # Handle real estate taxes
            elif key_lower in ["property_tax", "real_estate_tax", "e18500"]:
                tc_inputs["e18500"] = float(value)

            # Handle mortgage interest
            elif key_lower in ["mortgage_interest", "e19200"]:
                tc_inputs["e19200"] = float(value)

            # Handle charitable contributions
            elif key_lower in ["charitable", "charity", "donations", "e19800"]:
                tc_inputs["e19800"] = float(value)

        # Post-processing: If joint filing and no spouse age set, use primary age
        if tc_inputs["MARS"] == 2 and tc_inputs["age_spouse"] == 0:
            tc_inputs["age_spouse"] = tc_inputs["age_head"]

        return pd.DataFrame([tc_inputs])

    def validate(self, test_case: TestCase, variable: str, year: int = 2024) -> ValidatorResult:
        """Run validation using PSL Tax-Calculator.

        Args:
            test_case: The test case with inputs
            variable: The variable to calculate (e.g., "eitc", "income_tax")
            year: Tax year (2013 onwards supported)

        Returns:
            ValidatorResult with calculated value or error
        """
        tc = self._get_tc_module()

        # Map variable name
        var_lower = variable.lower()
        tc_variable = VARIABLE_MAPPING.get(var_lower, var_lower)

        if var_lower not in SUPPORTED_VARIABLES and tc_variable not in SUPPORTED_VARIABLES:
            return ValidatorResult(
                validator_name=self.name,
                validator_type=self.validator_type,
                calculated_value=None,
                error=f"Variable '{variable}' not supported by Tax-Calculator validator",
            )

        try:
            # Build input DataFrame
            input_df = self._build_input_dataframe(test_case, year)

            # Create Records object with custom data
            # We disable growth factors and weights for single-record calculation
            recs = tc.Records(
                data=input_df,
                start_year=year,
                gfactors=None,
                weights=None,
            )

            # Create Policy object (current law)
            pol = tc.Policy()

            # Create Calculator
            calc = tc.Calculator(policy=pol, records=recs)

            # Calculate all tax variables
            calc.calc_all()

            # Extract the requested variable
            result_df = calc.dataframe([tc_variable])
            calculated = float(result_df[tc_variable].iloc[0])

            return ValidatorResult(
                validator_name=self.name,
                validator_type=self.validator_type,
                calculated_value=calculated,
                metadata={
                    "tc_variable": tc_variable,
                    "year": year,
                    "input_mars": int(input_df["MARS"].iloc[0]),
                    "input_wages": float(input_df["e00200"].iloc[0]),
                },
            )

        except Exception as e:
            return ValidatorResult(
                validator_name=self.name,
                validator_type=self.validator_type,
                calculated_value=None,
                error=f"Tax-Calculator execution failed: {e}",
                metadata={"tc_variable": tc_variable, "year": year},
            )

    def get_all_outputs(self, test_case: TestCase, year: int = 2024) -> dict[str, float | None]:
        """Calculate all supported output variables for a test case.

        This is useful for comparing multiple variables at once without
        running the calculator multiple times.

        Args:
            test_case: The test case with inputs
            year: Tax year

        Returns:
            Dictionary mapping variable names to calculated values
        """
        tc = self._get_tc_module()

        try:
            # Build input DataFrame
            input_df = self._build_input_dataframe(test_case, year)

            # Create Records, Policy, and Calculator
            recs = tc.Records(
                data=input_df,
                start_year=year,
                gfactors=None,
                weights=None,
            )
            pol = tc.Policy()
            calc = tc.Calculator(policy=pol, records=recs)
            calc.calc_all()

            # Get unique Tax-Calculator variable names
            tc_vars = list(set(VARIABLE_MAPPING.values()))
            result_df = calc.dataframe(tc_vars)

            # Build output dictionary with common names
            outputs: dict[str, float | None] = {}
            for common_name, tc_var in VARIABLE_MAPPING.items():
                if tc_var in result_df.columns:
                    outputs[common_name] = float(result_df[tc_var].iloc[0])
                else:
                    outputs[common_name] = None

            return outputs

        except Exception:
            # Return None for all variables on error
            return {var: None for var in VARIABLE_MAPPING}
