"""Multi-validator comparison framework.

Compares Cosilico outputs against multiple external validators:
- PolicyEngine (primary - same microdata)
- TAXSIM (NBER reference calculator)
- Tax-Calculator (PSL microsimulation)

This isolates differences in rule implementations vs input data handling.
"""

import os
import platform
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from cosilico_validators.validators.base import TestCase
from cosilico_validators.validators.policyengine import PolicyEngineValidator
from cosilico_validators.validators.taxcalc import TaxCalculatorValidator
from cosilico_validators.validators.taxsim import TaxsimValidator

# TAXSIM executable download URLs (from policyengine-taxsim repo)
# These are bundled in https://github.com/PolicyEngine/policyengine-taxsim
TAXSIM_DOWNLOAD_URLS = {
    "darwin": "https://raw.githubusercontent.com/PolicyEngine/policyengine-taxsim/main/resources/taxsimtest/taxsimtest-osx.exe",
    "linux": "https://raw.githubusercontent.com/PolicyEngine/policyengine-taxsim/main/resources/taxsimtest/taxsimtest-linux.exe",
    "windows": "https://raw.githubusercontent.com/PolicyEngine/policyengine-taxsim/main/resources/taxsimtest/taxsimtest-windows.exe",
}


def get_taxsim_executable_path() -> Path:
    """Get path to TAXSIM executable, downloading if needed.

    Downloads from the policyengine-taxsim repository which bundles
    the TAXSIM executables for all platforms.
    """
    cache_dir = Path.home() / ".cache" / "cosilico-validators" / "taxsim"
    cache_dir.mkdir(parents=True, exist_ok=True)

    system = platform.system().lower()
    if system == "darwin":
        exe_name = "taxsimtest-osx.exe"
        url = TAXSIM_DOWNLOAD_URLS["darwin"]
    elif system == "linux":
        exe_name = "taxsimtest-linux.exe"
        url = TAXSIM_DOWNLOAD_URLS["linux"]
    else:
        exe_name = "taxsimtest-windows.exe"
        url = TAXSIM_DOWNLOAD_URLS["windows"]

    exe_path = cache_dir / exe_name

    if not exe_path.exists():
        print("Downloading TAXSIM from policyengine-taxsim repo...")
        print(f"  URL: {url}")
        urllib.request.urlretrieve(url, exe_path)
        # Make executable on Unix
        if system != "windows":
            os.chmod(exe_path, 0o755)
        print(f"  Saved to {exe_path}")

    return exe_path


@dataclass
class ValidatorComparison:
    """Comparison result for a single variable across multiple validators."""

    variable: str
    cosilico_value: float
    validator_results: dict[str, float | None]  # validator_name -> value
    differences: dict[str, float | None]  # validator_name -> difference
    match_flags: dict[str, bool]  # validator_name -> within tolerance


@dataclass
class MultiValidatorResult:
    """Results from comparing against multiple validators."""

    variable: str
    n_records: int
    validators_used: list[str]
    match_rates: dict[str, float]  # validator_name -> match rate
    mean_errors: dict[str, float]  # validator_name -> MAE
    weighted_totals: dict[str, float]  # source -> weighted total
    summary: dict = field(default_factory=dict)


def compare_single_case(
    test_case: TestCase,
    cosilico_value: float,
    variable: str,
    year: int = 2023,
    tolerance: float = 1.0,
    validators: Optional[list[str]] = None,
    taxsim_mode: str = "local",
) -> ValidatorComparison:
    """Compare a single test case against multiple validators.

    Args:
        test_case: Input test case
        cosilico_value: Cosilico's calculated value
        variable: Variable name (e.g., "eitc")
        year: Tax year
        tolerance: Match tolerance in dollars
        validators: List of validators to use ["policyengine", "taxsim", "taxcalc"]
        taxsim_mode: "local" (like policyengine-taxsim) or "web"

    Returns:
        ValidatorComparison with results from each validator
    """
    if validators is None:
        validators = ["policyengine", "taxsim", "taxcalc"]

    results = {}
    differences = {}
    match_flags = {}

    for validator_name in validators:
        try:
            if validator_name == "policyengine":
                validator = PolicyEngineValidator()
            elif validator_name == "taxsim":
                if taxsim_mode == "local":
                    exe_path = get_taxsim_executable_path()
                    validator = TaxsimValidator(mode="local", taxsim_path=exe_path)
                else:
                    validator = TaxsimValidator(mode="web")
            elif validator_name == "taxcalc":
                validator = TaxCalculatorValidator()
            else:
                continue

            result = validator.validate(test_case, variable, year)

            if result.success and result.calculated_value is not None:
                results[validator_name] = result.calculated_value
                diff = cosilico_value - result.calculated_value
                differences[validator_name] = diff
                match_flags[validator_name] = abs(diff) <= tolerance
            else:
                results[validator_name] = None
                differences[validator_name] = None
                match_flags[validator_name] = False

        except Exception as e:
            print(f"  Warning: {validator_name} failed: {e}")
            results[validator_name] = None
            differences[validator_name] = None
            match_flags[validator_name] = False

    return ValidatorComparison(
        variable=variable,
        cosilico_value=cosilico_value,
        validator_results=results,
        differences=differences,
        match_flags=match_flags,
    )


def compare_microdata(
    cosilico_values: np.ndarray,
    input_builder: Callable[[int], TestCase],
    variable: str,
    year: int = 2023,
    weights: Optional[np.ndarray] = None,
    tolerance: float = 1.0,
    validators: Optional[list[str]] = None,
    taxsim_mode: str = "local",
    sample_size: Optional[int] = None,
) -> MultiValidatorResult:
    """Compare Cosilico outputs against multiple validators on microdata.

    Args:
        cosilico_values: Array of Cosilico calculated values
        input_builder: Function that takes index and returns TestCase
        variable: Variable name (e.g., "eitc")
        year: Tax year
        weights: Optional weights for aggregation
        tolerance: Match tolerance in dollars
        validators: List of validators ["policyengine", "taxsim", "taxcalc"]
        taxsim_mode: "local" or "web" for TAXSIM
        sample_size: Optional limit for testing

    Returns:
        MultiValidatorResult with aggregate statistics
    """
    if validators is None:
        validators = ["taxsim", "taxcalc"]  # Skip PE since we use its inputs

    n = len(cosilico_values)
    if sample_size:
        n = min(n, sample_size)

    if weights is None:
        weights = np.ones(n)

    # Initialize validator instances
    validator_instances = {}
    for name in validators:
        try:
            if name == "taxsim":
                if taxsim_mode == "local":
                    exe_path = get_taxsim_executable_path()
                    validator_instances[name] = TaxsimValidator(mode="local", taxsim_path=exe_path)
                else:
                    validator_instances[name] = TaxsimValidator(mode="web")
            elif name == "taxcalc":
                validator_instances[name] = TaxCalculatorValidator()
            elif name == "policyengine":
                validator_instances[name] = PolicyEngineValidator()
        except Exception as e:
            print(f"Warning: Could not initialize {name}: {e}")

    # Build test cases for batch processing
    test_cases = [input_builder(i) for i in range(n)]

    # Collect results per validator
    validator_values: dict[str, list[float | None]] = {name: [] for name in validator_instances}

    print(f"Running {len(validator_instances)} validators on {n} records...")

    for name, validator in validator_instances.items():
        print(f"  {name}...", end=" ", flush=True)

        # Try batch validation if available
        if hasattr(validator, "batch_validate"):
            results = validator.batch_validate(test_cases, variable, year)
            for r in results:
                validator_values[name].append(r.calculated_value if r.success else None)
        else:
            # Fall back to single validation
            for tc in test_cases:
                result = validator.validate(tc, variable, year)
                validator_values[name].append(result.calculated_value if result.success else None)

        # Count successful results
        success_count = sum(1 for v in validator_values[name] if v is not None)
        print(f"{success_count}/{n} successful")

    # Compute match rates and errors
    match_rates = {}
    mean_errors = {}
    weighted_totals = {"cosilico": float((cosilico_values[:n] * weights[:n]).sum())}

    for name, values in validator_values.items():
        # Convert to array, handling None
        val_array = np.array([v if v is not None else np.nan for v in values], dtype=float)

        # Only compare where we have both values
        valid_mask = ~np.isnan(val_array)
        if valid_mask.sum() == 0:
            match_rates[name] = 0.0
            mean_errors[name] = float("nan")
            weighted_totals[name] = 0.0
            continue

        cos_valid = cosilico_values[:n][valid_mask]
        val_valid = val_array[valid_mask]
        weights_valid = weights[:n][valid_mask]

        # Match rate
        diffs = np.abs(cos_valid - val_valid)
        match_rates[name] = float((diffs <= tolerance).sum() / len(diffs))

        # MAE
        mean_errors[name] = float(diffs.mean())

        # Weighted totals
        weighted_totals[name] = float((val_valid * weights_valid).sum())

    return MultiValidatorResult(
        variable=variable,
        n_records=n,
        validators_used=list(validator_instances.keys()),
        match_rates=match_rates,
        mean_errors=mean_errors,
        weighted_totals=weighted_totals,
        summary={
            "cosilico_mean": float(cosilico_values[:n].mean()),
            "cosilico_total": weighted_totals["cosilico"],
        },
    )


def run_comparison_demo(year: int = 2023):
    """Run a demo comparison on a few test cases."""
    print("=" * 60)
    print("Multi-Validator Comparison Demo")
    print("=" * 60)

    # Sample test cases
    test_cases = [
        TestCase(
            name="Single, $20k wages, no kids",
            inputs={
                "earned_income": 20000,
                "filing_status": "SINGLE",
                "age": 30,
                "num_children": 0,
            },
            expected={},
        ),
        TestCase(
            name="Single, $15k wages, 2 kids",
            inputs={
                "earned_income": 15000,
                "filing_status": "SINGLE",
                "age": 30,
                "num_children": 2,
            },
            expected={},
        ),
        TestCase(
            name="Joint, $50k wages, 1 kid",
            inputs={
                "earned_income": 50000,
                "filing_status": "JOINT",
                "age": 35,
                "num_children": 1,
            },
            expected={},
        ),
    ]

    # Pretend Cosilico values (would come from engine)
    cosilico_eitc = [560.0, 5500.0, 0.0]

    print(f"\nComparing EITC for {len(test_cases)} test cases...")
    print("-" * 60)

    for i, tc in enumerate(test_cases):
        print(f"\n{tc.name}")
        print(f"  Cosilico EITC: ${cosilico_eitc[i]:,.2f}")

        result = compare_single_case(
            test_case=tc,
            cosilico_value=cosilico_eitc[i],
            variable="eitc",
            year=year,
            taxsim_mode="local",  # Use local like policyengine-taxsim
        )

        for validator_name, value in result.validator_results.items():
            if value is not None:
                diff = result.differences[validator_name]
                match = "✓" if result.match_flags[validator_name] else "✗"
                print(f"  {validator_name}: ${value:,.2f} (diff: ${diff:+,.2f}) {match}")
            else:
                print(f"  {validator_name}: Failed")


if __name__ == "__main__":  # pragma: no cover
    run_comparison_demo()
