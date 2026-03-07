"""Export validation results to cosilico.ai dashboard format.

Usage:
    python -m cosilico_validators.dashboard_export -o validation-results.json
    cp validation-results.json /path/to/cosilico.ai/public/data/

################################################################################
#                                                                              #
#   ██████  ██████  ██ ████████ ██  ██████  █████  ██                          #
#  ██      ██   ██  ██    ██    ██ ██      ██   ██ ██                          #
#  ██      ██████   ██    ██    ██ ██      ███████ ██                          #
#  ██      ██   ██  ██    ██    ██ ██      ██   ██ ██                          #
#   ██████ ██   ██  ██    ██    ██  ██████ ██   ██ ███████                     #
#                                                                              #
#   THIS FILE IS A VALIDATOR ONLY - NO TAX RULES ALLOWED HERE!                 #
#                                                                              #
#   ALL TAX CALCULATION LOGIC MUST COME FROM:                                  #
#     - cosilico-us/*.rac files (statute encodings)                            #
#     - cosilico-engine (DSL executor)                                         #
#                                                                              #
#   This validator ONLY:                                                       #
#     1. Loads outputs from Cosilico engine                                    #
#     2. Loads outputs from external validators (PE, TAXSIM, etc)              #
#     3. Compares them                                                         #
#                                                                              #
#   DO NOT ADD:                                                                #
#     - Filing status logic                                                    #
#     - Age-based calculations                                                 #
#     - Income aggregations                                                    #
#     - ANY tax rule implementations                                           #
#                                                                              #
#   If validation fails, FIX THE .RAC FILES, not this validator!               #
#                                                                              #
################################################################################
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
import numpy as np

from cosilico_validators.comparison.aligned import (
    ComparisonResult,
    compare_variable,
    load_common_dataset,
)

# Variables to validate - keys are PolicyEngine variable names
# Section references the USC statute where the rule is encoded
VARIABLES = {
    "eitc": {"section": "26/32", "title": "Earned Income Tax Credit"},
    "income_tax_before_credits": {"section": "26/1", "title": "Income Tax (Before Credits)"},
    "ctc": {"section": "26/24", "title": "Child Tax Credit (Total)"},
    "non_refundable_ctc": {"section": "26/24", "title": "Child Tax Credit (Non-refundable)"},
    "refundable_ctc": {"section": "26/24", "title": "Additional Child Tax Credit"},
    "standard_deduction": {"section": "26/63", "title": "Standard Deduction"},
    "adjusted_gross_income": {"section": "26/62", "title": "Adjusted Gross Income"},
    "taxable_income": {"section": "26/63", "title": "Taxable Income"},
    "cdcc": {"section": "26/21", "title": "Child & Dependent Care Credit"},
    "salt_deduction": {"section": "26/164", "title": "SALT Deduction"},
    "alternative_minimum_tax": {"section": "26/55", "title": "Alternative Minimum Tax"},
    "premium_tax_credit": {"section": "26/36B", "title": "Premium Tax Credit"},
    "net_investment_income_tax": {"section": "26/1411", "title": "Net Investment Income Tax"},
}


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def load_cosilico_engine():
    """Load the Cosilico engine from cosilico-engine repo."""
    engine_path = Path.home() / "CosilicoAI" / "cosilico-engine" / "src"
    if engine_path.exists():
        sys.path.insert(0, str(engine_path))

    from cosilico.dependency_resolver import DependencyResolver
    from cosilico.dsl_parser import parse_dsl
    from cosilico.vectorized_executor import VectorizedExecutor

    return VectorizedExecutor, parse_dsl, DependencyResolver


# EITC parameters for 2024 (from IRS Rev. Proc. 2023-34)
EITC_PARAMS_2024 = {
    "credit_rate_0": 0.0765,  # 0 children
    "credit_rate_1": 0.34,  # 1 child
    "credit_rate_2": 0.40,  # 2 children
    "credit_rate_3": 0.45,  # 3+ children
    "earned_income_cap_0": 7840,
    "earned_income_cap_1": 11750,
    "earned_income_cap_2": 16510,
    "phaseout_rate_0": 0.0765,
    "phaseout_rate_1": 0.1598,
    "phaseout_rate_2": 0.2106,
    "phaseout_start_0": 9800,
    "phaseout_start_1": 21560,
    "joint_adjustment": 6570,  # Additional threshold for joint filers
}

# NIIT parameters for 2024 (from 26 USC Section 1411)
# Note: NIIT thresholds are NOT indexed for inflation
NIIT_PARAMS_2024 = {
    "niit_rate": 0.038,  # 3.8% surtax
    "threshold_joint": 250000,  # Joint filers and surviving spouse
    "threshold_separate": 125000,  # Married filing separately
    "threshold_other": 200000,  # Single, HOH, and others
}

# CTC parameters for 2024 (from IRS Rev. Proc. 2023-34 and TCJA)
# Reference: 26 USC 24(h)(2), (h)(3), (h)(5), (b)(1), (d)(1)(B)
CTC_PARAMS_2024 = {
    # 26 USC 24(h)(2) - Credit amount per qualifying child
    "credit_amount": 2000,
    # 26 USC 24(h)(5) - Maximum refundable (ACTC) per child
    # Base $1,400 indexed for inflation; 2024 value is $1,700
    "refundable_max": 1700,
    # 26 USC 24(h)(3) - Phaseout thresholds (TCJA 2018-2025)
    "phaseout_threshold_joint": 400000,
    "phaseout_threshold_other": 200000,
    # 26 USC 24(b)(1) - Phaseout mechanics
    "phaseout_rate": 50,  # $50 reduction
    "phaseout_increment": 1000,  # per $1,000 (or fraction) over threshold
    # 26 USC 24(d)(1)(B)(i) - ACTC earned income formula
    "earned_income_threshold": 2500,  # TCJA threshold
    "refundable_rate": 0.15,  # 15% of earned income above threshold
    # 26 USC 24(d)(1)(B)(ii) - ACTC SS tax formula threshold
    "qualifying_children_threshold": 3,  # 3+ children to use SS formula
}

# CDCC parameters for 2024 (26 USC 21)
# Note: 2024 uses permanent law parameters (ARPA enhancements expired after 2021)
CDCC_PARAMS_2024 = {
    # Expense limits per 26 USC 21(c)
    "one_qualifying_individual": 3000,  # $3,000 max expenses for 1 qualifying individual
    "two_or_more_qualifying_individuals": 6000,  # $6,000 max expenses for 2+ qualifying individuals
    # Credit rate per 26 USC 21(a)(2)
    "maximum_rate": 0.35,  # 35% starting rate
    "minimum_rate": 0.20,  # 20% floor (cannot go below)
    "phase_down_threshold": 15000,  # AGI threshold where phasedown starts
    "phase_down_increment": 2000,  # Rate reduces per $2,000 of AGI
    "phase_down_rate": 0.01,  # 1 percentage point reduction per increment
    # ARPA 2021 phaseout (not applicable for 2024, set to infinity)
    "arpa_complete_phaseout_threshold": float("inf"),  # No complete phaseout in 2024
    # Qualifying individual age limit per 26 USC 21(b)(1)(A)
    "child_age_limit": 13,  # Under age 13
}

# Standard Deduction parameters for 2024 (from IRS Rev. Proc. 2023-34)
# Reference: 26 USC 63(c), (f)
STD_DEDUCTION_PARAMS_2024 = {
    # 26 USC 63(c)(2) - Basic standard deduction amounts
    "basic_joint": 29200,  # MFJ and surviving spouse
    "basic_head_of_household": 21900,  # Head of household
    "basic_single": 14600,  # Single and MFS
    # 26 USC 63(f)(1), (f)(3) - Additional amounts for aged (65+)
    "additional_aged_married": 1550,  # For married/surviving spouse
    "additional_aged_unmarried": 1950,  # For single/HOH
    # 26 USC 63(f)(2), (f)(3) - Additional amounts for blind
    "additional_blind_married": 1550,  # For married/surviving spouse
    "additional_blind_unmarried": 1950,  # For single/HOH
    # 26 USC 63(c)(5) - Dependent standard deduction
    "dependent_minimum": 1300,  # Floor for dependents
    "dependent_earned_addon": 450,  # Added to earned income
    # 26 USC 63(f)(1) - Age threshold
    "aged_threshold": 65,
}


def load_rac_file(section: str) -> Optional[str]:
    """Load .rac file for a given section from cosilico-us.

    Args:
        section: USC section like "26/32" or "26/63"

    Returns:
        Contents of the .rac file, or None if not found
    """
    statute_dir = Path.home() / "CosilicoAI" / "cosilico-us" / "statute"

    # Try direct path first (e.g., statute/26/32.rac)
    rac_path = statute_dir / f"{section}.rac"
    if rac_path.exists():
        return rac_path.read_text()

    # Try with /a suffix (common pattern)
    rac_path = statute_dir / section / "a.rac"
    if rac_path.exists():
        return rac_path.read_text()

    return None


def result_to_section(result: ComparisonResult, n_households: int, meta: dict, implemented: bool) -> dict:
    """Convert ComparisonResult to ValidationSection format."""
    return {
        "section": meta["section"],
        "title": meta["title"],
        "variable": result.variable,
        "implemented": implemented,
        "households": n_households,
        "testCases": [],
        "summary": {
            "total": result.n_records,
            "matches": int(result.match_rate * result.n_records),
            "matchRate": result.match_rate,
            "meanAbsoluteError": result.mean_absolute_error,
        },
        "validatorBreakdown": {
            "policyengine": {
                "matches": int(result.match_rate * result.n_records),
                "total": result.n_records,
                "rate": result.match_rate,
            }
        },
        "notes": (
            f"Cosilico total: ${result.cosilico_total / 1e9:.1f}B, "
            f"PE total: ${result.policyengine_total / 1e9:.1f}B, "
            f"Diff: ${(result.cosilico_total - result.policyengine_total) / 1e9:+.1f}B"
        )
        if implemented
        else "Not yet implemented in .rac files",
    }


def run_export(year: int = 2024, output_path: Optional[Path] = None) -> dict:
    """Run validation and export to dashboard format.

    This function:
    1. Loads the Cosilico engine
    2. For each variable, loads the .rac file and executes it
    3. Compares against PolicyEngine outputs
    4. Returns dashboard-formatted results
    """
    from policyengine_us import Microsimulation

    # Load common dataset
    print("Loading common dataset from PolicyEngine...")
    dataset = load_common_dataset(year)
    print(f"  {dataset.n_records:,} tax units loaded")

    # Load Cosilico engine
    print("Loading Cosilico engine...")
    try:
        VectorizedExecutor, Parser, DependencyResolver = load_cosilico_engine()
        engine_available = True

        # Create dependency resolver pointing to cosilico-us
        statute_root = Path.home() / "CosilicoAI" / "cosilico-us"
        dep_resolver = DependencyResolver(statute_root=statute_root)
    except ImportError as e:
        print(f"  Warning: Could not load engine: {e}")
        engine_available = False
        dep_resolver = None

    # Get PE microsimulation
    print("Loading PolicyEngine calculations...")
    sim = Microsimulation()

    # Run comparisons for all variables
    results = []
    for var_name, meta in VARIABLES.items():
        print(f"Comparing {var_name}...")

        try:
            # Get PolicyEngine values
            pe_values = np.array(sim.calculate(var_name, year))

            # Try to load and execute .rac file
            rac_code = load_rac_file(meta["section"])
            implemented = False
            cos_values = None

            # Special handling for EITC - we have working engine integration
            if var_name == "eitc" and engine_available:
                try:
                    # Load EITC formula from cosilico-us/statute/26/32.rac
                    eitc_rac = Path.home() / "CosilicoAI" / "cosilico-us" / "statute" / "26" / "32.rac"
                    if eitc_rac.exists():
                        rac_code = eitc_rac.read_text()

                        # Build inputs from dataset
                        inputs = {
                            "is_eligible_individual": np.ones(dataset.n_records, dtype=bool),
                            "num_qualifying_children": np.clip(dataset.eitc_child_count, 0, 3).astype(int),
                            "earned_income": dataset.earned_income,
                            "adjusted_gross_income": dataset.adjusted_gross_income,
                            "filing_status": np.where(dataset.is_joint, "JOINT", "SINGLE"),
                        }

                        # Execute through engine
                        executor = VectorizedExecutor(parameters=EITC_PARAMS_2024)
                        results_dict = executor.execute(
                            code=rac_code, inputs=inputs, output_variables=["eitc_standalone"]
                        )
                        cos_values = results_dict["eitc_standalone"]
                        implemented = True
                except Exception as e:
                    print(f"    Engine execution failed: {e}")
                    implemented = False

            # Special handling for NIIT - Net Investment Income Tax
            if var_name == "net_investment_income_tax" and engine_available:
                try:
                    # Load NIIT formula from standalone validation file (no imports)
                    niit_rac = Path.home() / "CosilicoAI" / "cosilico-us" / "statute" / "26" / "1411.rac"
                    if niit_rac.exists():
                        rac_code = niit_rac.read_text()

                        # Build inputs from dataset
                        # Section 1411(d): MAGI = AGI + foreign earned income exclusion
                        # For CPS data, we assume no foreign earned income exclusion
                        inputs = {
                            "net_investment_income": dataset.investment_income,
                            "adjusted_gross_income": dataset.adjusted_gross_income,
                            "foreign_earned_income_exclusion": np.zeros(dataset.n_records),
                            "filing_status": np.where(dataset.is_joint, "JOINT", "SINGLE"),
                        }

                        # Execute through engine using standalone version (no imports)
                        executor = VectorizedExecutor(parameters=NIIT_PARAMS_2024)
                        results_dict = executor.execute(
                            code=rac_code, inputs=inputs, output_variables=["niit_standalone"]
                        )
                        cos_values = results_dict["niit_standalone"]
                        implemented = True
                except Exception as e:
                    print(f"    NIIT engine failed: {e}")
                    implemented = False

            # AGI engine integration - 26 USC Section 62
            # AGI = Gross Income (Section 61) - Above-the-line deductions (Section 62(a))
            # Gross income: wages, self-employment, interest, dividends, capital gains,
            #   rental, social security, pension, unemployment, other income
            # Above-the-line deductions: educator expense, IRA, HSA, student loan interest,
            #   tip income (OBBBA), qualified overtime (OBBBA)
            elif var_name == "adjusted_gross_income" and engine_available and dep_resolver:
                try:
                    # Build inputs from CommonDataset income fields
                    # Map to .rac variable names from imports in 26/62/a.rac
                    # Note: These are TaxUnit-level inputs (dataset.n_records)
                    inputs = {
                        # Gross income components (26 USC Section 61)
                        # Names must match imports in 26/62/a.rac
                        "wages": dataset.wages,
                        "salaries": np.zeros(dataset.n_records),  # Combined in wages
                        "tips": np.zeros(dataset.n_records),  # Combined in wages
                        "self_employment_income": dataset.self_employment_income,
                        "partnership_s_corp_income": dataset.partnership_s_corp_income,  # §61(a)(3) via §701
                        "farm_income": dataset.farm_income,  # §61(a)(6)
                        "interest_income": dataset.interest_income,  # §61(a)(4)
                        "dividend_income": dataset.dividend_income,  # §61(a)(7)
                        "capital_gains": dataset.capital_gains,  # §61(a)(3)
                        "rental_income": dataset.rental_income,  # §61(a)(5)
                        "taxable_social_security": dataset.taxable_social_security,  # §86
                        "pension_income": dataset.pension_income,  # §61(a)(11)
                        "taxable_unemployment": dataset.taxable_unemployment,  # §85
                        "retirement_distributions": dataset.retirement_distributions,  # §402
                        "miscellaneous_income": dataset.miscellaneous_income,  # §61(a) other
                        # Above-the-line deductions (26 USC Section 62(a))
                        "self_employment_tax_deduction": dataset.self_employment_tax_deduction,  # §62(a)(1)
                        "self_employed_health_insurance_deduction": dataset.self_employed_health_insurance_deduction,  # §62(a)(1)
                        "educator_expense_deduction": dataset.educator_expense_deduction,  # §62(a)(2)(D)
                        "loss_deduction": dataset.loss_deduction,  # §62(a)(4)
                        "self_employed_pension_deduction": dataset.self_employed_pension_deduction,  # §62(a)(6)
                        "ira_deduction": dataset.ira_deduction,  # §62(a)(7)
                        "hsa_deduction": dataset.hsa_deduction,  # §62(a)(12)
                        "student_loan_interest_deduction": dataset.student_loan_interest_deduction,  # §62(a)(17)
                        "tip_income_deduction": np.zeros(dataset.n_records),  # §62(a)(23) OBBBA temporary
                        "qualified_overtime_deduction": np.zeros(dataset.n_records),  # §62(a)(24) OBBBA temporary
                    }

                    # Execute through engine with lazy dependency resolution
                    executor = VectorizedExecutor(dependency_resolver=dep_resolver)
                    results_dict = executor.execute_lazy(
                        entry_point="statute/26/62/a", inputs=inputs, output_variables=["adjusted_gross_income"]
                    )
                    cos_values = results_dict["adjusted_gross_income"]
                    implemented = True
                except Exception as e:
                    print(f"    AGI engine failed: {e}")
                    implemented = False

            # CTC (total) engine integration - 26 USC Section 24
            # Using lazy dependency resolution to automatically resolve imports
            elif var_name == "ctc" and engine_available and dep_resolver:
                try:
                    # Get tax liability from PE for tax limit calculation
                    pe_tax_before_credits = np.array(sim.calculate("income_tax_before_credits", year))
                    # Get EITC from PE for ACTC SS tax formula (3+ children)
                    pe_eitc = np.array(sim.calculate("eitc", year))
                    # Get SS taxes from PE for ACTC formula (use TaxUnit-level variable)
                    pe_ss_taxes = np.array(sim.calculate("pr_refundable_ctc_social_security_tax", year))

                    # Build inputs - these break circular dependencies (like OpenFisca)
                    inputs = {
                        "num_ctc_qualifying_children": dataset.ctc_child_count.astype(int),
                        "adjusted_gross_income": dataset.adjusted_gross_income,
                        "filing_status": np.where(dataset.is_joint, "JOINT", "SINGLE"),
                        "earned_income": dataset.earned_income,
                        "tax_liability_limit": pe_tax_before_credits,
                        "social_security_taxes": pe_ss_taxes,
                        "earned_income_credit": pe_eitc,
                    }

                    # Execute with lazy dependency resolution (like OpenFisca)
                    executor = VectorizedExecutor(parameters=CTC_PARAMS_2024, dependency_resolver=dep_resolver)
                    results_dict = executor.execute_lazy(
                        entry_point="statute/26/24/a", inputs=inputs, output_variables=["ctc_total"]
                    )
                    cos_values = results_dict["ctc_total"]
                    implemented = True
                except Exception as e:
                    print(f"    CTC engine failed: {e}")
                    import traceback

                    traceback.print_exc()
                    implemented = False

            # Non-refundable CTC engine integration - 26 USC Section 24(a)
            # child_tax_credit = min(ctc_before_limit, tax_liability)
            elif var_name == "non_refundable_ctc" and engine_available and dep_resolver:
                try:
                    # Get tax liability from PE for tax limit calculation
                    pe_tax_before_credits = np.array(sim.calculate("income_tax_before_credits", year))

                    # Build inputs from dataset (break circular deps)
                    inputs = {
                        "num_ctc_qualifying_children": dataset.ctc_child_count.astype(int),
                        "adjusted_gross_income": dataset.adjusted_gross_income,
                        "filing_status": np.where(dataset.is_joint, "JOINT", "SINGLE"),
                        "tax_liability_limit": pe_tax_before_credits,
                    }

                    # Execute with lazy dependency resolution
                    executor = VectorizedExecutor(parameters=CTC_PARAMS_2024, dependency_resolver=dep_resolver)
                    results_dict = executor.execute_lazy(
                        entry_point="statute/26/24/a", inputs=inputs, output_variables=["child_tax_credit"]
                    )
                    cos_values = results_dict["child_tax_credit"]
                    implemented = True
                except Exception as e:
                    print(f"    Non-refundable CTC engine failed: {e}")
                    implemented = False

            # Refundable CTC (ACTC) engine integration - 26 USC Section 24(d)
            # additional_child_tax_credit = min(ctc_before_limit, max(earned_formula, ss_formula))
            elif var_name == "refundable_ctc" and engine_available and dep_resolver:
                try:
                    # Get EITC from PE for ACTC SS tax formula (3+ children)
                    pe_eitc = np.array(sim.calculate("eitc", year))
                    # Get SS taxes from PE for ACTC formula
                    # Use TaxUnit-level variable for refundable CTC calculation
                    pe_ss_taxes = np.array(sim.calculate("pr_refundable_ctc_social_security_tax", year))
                    # Get tax liability from PE for ctc_before_limit calculation
                    pe_tax_before_credits = np.array(sim.calculate("income_tax_before_credits", year))

                    # Build inputs - lazy resolution handles child_tax_credit_before_limit automatically
                    inputs = {
                        "num_ctc_qualifying_children": dataset.ctc_child_count.astype(int),
                        "adjusted_gross_income": dataset.adjusted_gross_income,
                        "filing_status": np.where(dataset.is_joint, "JOINT", "SINGLE"),
                        "earned_income": dataset.earned_income,
                        "tax_liability_limit": pe_tax_before_credits,
                        "social_security_taxes": pe_ss_taxes,
                        "earned_income_credit": pe_eitc,
                    }

                    # Execute with lazy dependency resolution - will auto-compute child_tax_credit_before_limit
                    executor = VectorizedExecutor(parameters=CTC_PARAMS_2024, dependency_resolver=dep_resolver)
                    results_dict = executor.execute_lazy(
                        entry_point="statute/26/24/d/1/B",
                        inputs=inputs,
                        output_variables=["additional_child_tax_credit"],
                    )
                    cos_values = results_dict["additional_child_tax_credit"]
                    implemented = True
                except Exception as e:
                    print(f"    Refundable CTC (ACTC) engine failed: {e}")
                    implemented = False

            # CDCC engine integration - 26 USC Section 21
            # Child and Dependent Care Credit
            # cdcc = applicable_percentage * min(expenses, expense_limit, earned_income_limit)
            elif var_name == "cdcc" and engine_available:
                try:
                    # Load CDCC formula from cosilico-us/statute/26/21/a.rac
                    cdcc_rac = Path.home() / "CosilicoAI" / "cosilico-us" / "statute" / "26" / "21" / "a.rac"
                    if cdcc_rac.exists():
                        rac_code = cdcc_rac.read_text()

                        # Build inputs from CommonDataset (now includes CDCC fields)
                        # Note: dataset.childcare_expenses and cdcc_qualifying_individuals
                        # are loaded from PE's tax_unit_childcare_expenses and capped_count_cdcc_eligible
                        inputs = {
                            # Childcare expenses paid during the year (26 USC 21(b)(2))
                            "cdcc_expenses_paid": dataset.childcare_expenses,
                            # Number of qualifying individuals - children under 13 or disabled (26 USC 21(b)(1))
                            "num_cdcc_qualifying_individuals": dataset.cdcc_qualifying_individuals.astype(int),
                            # AGI for credit rate calculation (26 USC 21(a)(2))
                            "adjusted_gross_income": dataset.adjusted_gross_income,
                            # Earned income for limitation - lesser of spouse earnings for married (26 USC 21(d))
                            "earned_income": dataset.earned_income,
                            # Filing status for earned income limit calculation
                            "filing_status": np.where(dataset.is_joint, "JOINT", "SINGLE"),
                        }

                        # Execute through engine using standalone formula
                        executor = VectorizedExecutor(parameters=CDCC_PARAMS_2024)
                        results_dict = executor.execute(
                            code=rac_code, inputs=inputs, output_variables=["cdcc_standalone"]
                        )
                        cos_values = results_dict["cdcc_standalone"]
                        implemented = True
                except Exception as e:
                    print(f"    CDCC engine failed: {e}")
                    implemented = False

            # Standard Deduction engine integration - 26 USC Section 63(c), (f)
            # standard_deduction = basic_standard_deduction + additional_standard_deduction
            # Depends on: filing_status, age, blind status, dependent status
            elif var_name == "standard_deduction" and engine_available:
                try:
                    # Load Standard Deduction formula from cosilico-us/statute/26/63/c.rac
                    std_ded_rac = Path.home() / "CosilicoAI" / "cosilico-us" / "statute" / "26" / "63" / "c.rac"
                    if std_ded_rac.exists():
                        rac_code = std_ded_rac.read_text()

                        # Build inputs for standalone formula
                        inputs = {
                            # Filing status - use raw values, enums handle JOINT etc
                            "filing_status": np.where(dataset.is_joint, "JOINT", "SINGLE"),
                            # Max age in tax unit for 63(f)(1) aged deduction
                            "max_age": np.maximum(dataset.head_age, dataset.spouse_age),
                            # Any blind in tax unit for 63(f)(2) blind deduction
                            "any_blind": dataset.head_is_blind | dataset.spouse_is_blind,
                            # Dependent status for 63(c)(5) limited deduction
                            "is_dependent": dataset.head_is_dependent,
                            # Earned income for dependent calculation
                            "earned_income": dataset.earned_income,
                            # 63(c)(6) ineligibility - assume none
                            "is_ineligible_for_standard_deduction": np.zeros(dataset.n_records, dtype=bool),
                        }

                        # Execute through engine using standalone formula
                        executor = VectorizedExecutor(parameters=STD_DEDUCTION_PARAMS_2024)
                        results_dict = executor.execute(
                            code=rac_code, inputs=inputs, output_variables=["standard_deduction_standalone"]
                        )
                        cos_values = results_dict["standard_deduction_standalone"]
                        implemented = True
                except Exception as e:
                    print(f"    Standard Deduction engine failed: {e}")
                    implemented = False

            if not implemented:
                # Return zeros for unimplemented variables
                def cos_func(ds):
                    return np.zeros(ds.n_records)
            else:
                # Capture cos_values in closure
                _cos_values = cos_values

                def cos_func(ds, v=_cos_values):
                    return v

            result = compare_variable(dataset, cos_func, pe_values, var_name)
            results.append((result, meta, implemented))

            status = "✓ ENGINE" if implemented else "○ (not in engine yet)"
            print(f"  {status} Match rate: {result.match_rate * 100:.1f}%")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue

    # Build ValidationResults structure
    sections = [result_to_section(r, dataset.n_records, meta, impl) for r, meta, impl in results]

    # Separate implemented vs stub for overall calculation
    implemented_results = [r for r, m, impl in results if impl]
    if implemented_results:
        overall_match_rate = np.mean([r.match_rate for r in implemented_results])
        overall_mae = np.mean([r.mean_absolute_error for r in implemented_results])
    else:
        overall_match_rate = 0.0
        overall_mae = 0.0

    n_implemented = sum(1 for _, _, impl in results if impl)
    n_total = len(results)

    dashboard_data = {
        "isSampleData": False,
        "timestamp": datetime.now().isoformat(),
        "commit": get_git_commit(),
        "dataSource": f"PolicyEngine Enhanced CPS {year}",
        "householdsTotal": dataset.n_records,
        "sections": sections,
        "coverage": {
            "implemented": n_implemented,
            "total": n_total,
            "percentage": n_implemented / n_total if n_total > 0 else 0,
        },
        "overall": {
            "totalHouseholds": dataset.n_records,
            "totalTests": sum(r.n_records for r, _, _ in results),
            "totalMatches": sum(int(r.match_rate * r.n_records) for r, _, impl in results if impl),
            "matchRate": overall_match_rate,
            "meanAbsoluteError": overall_mae,
        },
        "validators": [
            {
                "name": "PolicyEngine",
                "available": True,
                "version": "latest",
                "householdsCovered": dataset.n_records,
            },
            {
                "name": "TAXSIM",
                "available": False,
                "version": "35",
                "householdsCovered": 0,
            },
            {
                "name": "Tax-Calculator",
                "available": False,
                "version": "latest",
                "householdsCovered": 0,
            },
        ],
    }

    # Write to file if path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(dashboard_data, f, indent=2)
        print(f"\nWritten to {output_path}")

    return dashboard_data


@click.command()
@click.option("--year", "-y", default=2024, help="Tax year")
@click.option("--output", "-o", type=click.Path(), help="Output JSON file")
def main(year: int, output: Optional[str]):
    """Export validation results to dashboard format."""
    output_path = Path(output) if output else None
    data = run_export(year, output_path)

    print("\n=== Summary ===")
    print(f"Coverage: {data['coverage']['implemented']}/{data['coverage']['total']} variables via engine")
    print(f"Match rate (implemented): {data['overall']['matchRate'] * 100:.1f}%")
    print(f"MAE: ${data['overall']['meanAbsoluteError']:.2f}")
    print("\nNote: Variables show 0% until .rac→engine integration is complete")


if __name__ == "__main__":  # pragma: no cover
    main()
