"""Yale Budget Lab Tax-Simulator validator.

Uses the Yale Budget Lab's Tax-Simulator, an R-based microsimulation model
for analyzing U.S. federal tax policy.

See: https://github.com/Budget-Lab-Yale/Tax-Simulator
"""

import contextlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from cosilico_validators.validators.base import (
    BaseValidator,
    TestCase,
    ValidatorResult,
    ValidatorType,
)

# Variable mapping from common names to Yale Tax-Simulator output column names
# These are based on the Tax-Simulator's output format
VARIABLE_MAPPING = {
    # Income tax
    "income_tax": "income_tax",
    "federal_income_tax": "income_tax",
    "iitax": "income_tax",
    # EITC
    "eitc": "eitc",
    "earned_income_credit": "eitc",
    # Child Tax Credit
    "ctc": "ctc",
    "child_tax_credit": "ctc",
    # AGI
    "agi": "agi",
    "adjusted_gross_income": "agi",
    # Standard deduction
    "standard_deduction": "standard_deduction",
    # Taxable income
    "taxable_income": "taxable_income",
    # Other common variables
    "amt": "amt",
    "payroll_tax": "payroll_tax",
}

# Supported variables for this validator
SUPPORTED_VARIABLES = set(VARIABLE_MAPPING.keys()) | set(VARIABLE_MAPPING.values())

# Filing status mapping to Yale format
FILING_STATUS_MAP = {
    "SINGLE": "single",
    "JOINT": "married",
    "MARRIED_FILING_JOINTLY": "married",
    "MARRIED_FILING_SEPARATELY": "married_separate",
    "HEAD_OF_HOUSEHOLD": "head_of_household",
    "SEPARATE": "married_separate",
}


class YaleTaxValidator(BaseValidator):
    """Validator using Yale Budget Lab's Tax-Simulator.

    The Tax-Simulator is an R-based microsimulation model. This validator
    interfaces with it by:
    1. Creating temporary input files matching the simulator's expected format
    2. Running the R script via subprocess
    3. Parsing the output files

    Prerequisites:
    - R 4.0+ installed and in PATH
    - Tax-Simulator repository cloned locally
    - Required R packages installed (tidyverse, yaml, data.table, etc.)
    """

    name = "Yale Tax-Simulator"
    validator_type = ValidatorType.SUPPLEMENTARY
    supported_variables = SUPPORTED_VARIABLES

    def __init__(self, tax_simulator_path: str | Path | None = None):
        """Initialize Yale Tax-Simulator validator.

        Args:
            tax_simulator_path: Path to the cloned Tax-Simulator repository.
                If not provided, searches standard locations.
        """
        self.tax_simulator_path = self._resolve_path(tax_simulator_path)
        self._check_r_available()

    def _resolve_path(self, provided_path: str | Path | None) -> Path:
        """Find the Tax-Simulator repository."""
        if provided_path:
            path = Path(provided_path)
            if path.exists() and (path / "src" / "main.R").exists():
                return path
            raise FileNotFoundError(f"Tax-Simulator not found at: {path}. Expected to find src/main.R")

        # Search standard locations
        search_paths = [
            Path.home() / "Tax-Simulator",
            Path.home() / "repos" / "Tax-Simulator",
            Path.home() / "github" / "Budget-Lab-Yale" / "Tax-Simulator",
            Path.home() / ".cosilico" / "Tax-Simulator",
            Path.cwd() / "Tax-Simulator",
            Path.cwd().parent / "Tax-Simulator",
        ]

        for path in search_paths:
            if path.exists() and (path / "src" / "main.R").exists():
                return path

        raise FileNotFoundError(
            "Yale Tax-Simulator repository not found. "
            "Clone from https://github.com/Budget-Lab-Yale/Tax-Simulator "
            "to one of:\n" + "\n".join(f"  - {p}" for p in search_paths)
        )

    def _check_r_available(self) -> None:
        """Check that R and Rscript are available."""
        try:
            result = subprocess.run(
                ["Rscript", "--version"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError("Rscript returned non-zero exit code")
        except FileNotFoundError as e:
            raise RuntimeError("Rscript not found in PATH. Install R 4.0+ from https://cran.r-project.org/") from e

    def supports_variable(self, variable: str) -> bool:
        """Check if this validator supports a given variable."""
        return variable.lower() in SUPPORTED_VARIABLES

    def _create_tax_unit_input(self, test_case: TestCase, year: int, temp_dir: Path) -> Path:
        """Create a tax unit input file for the Tax-Simulator.

        The Tax-Simulator expects microdata in a specific format.
        This creates a minimal single-record input file.
        """
        inputs = test_case.inputs

        # Build the tax unit record
        record = {
            "RECID": 1,
            "FLPDYR": year,
            # Filing status
            "MARS": self._map_filing_status(inputs.get("filing_status", "SINGLE")),
            # Wages
            "e00200": inputs.get("earned_income", inputs.get("wages", 0)),
            "e00200p": inputs.get("earned_income", inputs.get("wages", 0)),
            "e00200s": inputs.get("spouse_wages", 0),
            # Ages
            "age_head": inputs.get("age", 30),
            "age_spouse": inputs.get("spouse_age", 0),
            # Children
            "n24": inputs.get("eitc_qualifying_children_count", inputs.get("num_children", 0)),
            "nu18": inputs.get("num_children", 0),
            # Weighting (single unit)
            "s006": 1.0,
        }

        # Handle additional income sources
        if "self_employment_income" in inputs:
            record["e00900"] = inputs["self_employment_income"]
        if "business_income" in inputs:
            record["e00900"] = inputs.get("e00900", 0) + inputs["business_income"]
        if "dividend_income" in inputs:
            record["e00600"] = inputs["dividend_income"]
        if "interest_income" in inputs:
            record["e00300"] = inputs["interest_income"]
        if "capital_gains" in inputs:
            record["p23250"] = inputs["capital_gains"]

        # Write as CSV
        input_file = temp_dir / "tax_units.csv"
        import csv

        with open(input_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=record.keys())
            writer.writeheader()
            writer.writerow(record)

        return input_file

    def _map_filing_status(self, status: str) -> int:
        """Map filing status string to MARS code.

        MARS values:
        1 = Single
        2 = Married filing jointly
        3 = Married filing separately
        4 = Head of household
        5 = Qualifying widow(er)
        """
        status_upper = status.upper() if isinstance(status, str) else "SINGLE"
        mapping = {
            "SINGLE": 1,
            "JOINT": 2,
            "MARRIED_FILING_JOINTLY": 2,
            "MARRIED_FILING_SEPARATELY": 3,
            "SEPARATE": 3,
            "HEAD_OF_HOUSEHOLD": 4,
            "WIDOW": 5,
            "WIDOWER": 5,
        }
        return mapping.get(status_upper, 1)

    def _create_runscript(self, year: int, temp_dir: Path) -> Path:
        """Create a runscript CSV for the Tax-Simulator."""
        runscript_file = temp_dir / "runscript.csv"

        # Minimal runscript for single-year calculation
        import csv

        with open(runscript_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["scenario_id", "tax_law", "behavior_module", "start_year", "end_year"])
            writer.writerow(["baseline", "current_law", "", str(year), str(year)])

        return runscript_file

    def _run_simulator(self, input_file: Path, runscript: Path, temp_dir: Path, year: int) -> dict[str, float]:
        """Run the Tax-Simulator and parse results.

        Note: This is a simplified implementation. The full Tax-Simulator
        expects properly formatted microdata and configuration files.
        For accurate validation, the Tax-Simulator may need custom
        integration code.
        """
        # The Tax-Simulator is complex and expects specific input formats.
        # For now, we'll attempt to run it but this may require
        # further customization based on the actual repo structure.

        output_dir = temp_dir / "output"
        output_dir.mkdir(exist_ok=True)

        # Build command
        cmd = [
            "Rscript",
            str(self.tax_simulator_path / "src" / "main.R"),
            str(runscript),  # runscript
            "1",  # scenario_id
            "cosilico",  # user_id
            "1",  # local mode
            str(year),  # vintage/year
            "100",  # pct_sample (100% for single unit)
            "0",  # stacked
            str(year),  # baseline_vintage
            "0",  # delete_detail
            "none",  # multicore
        ]

        env = os.environ.copy()
        env["TAX_SIMULATOR_INPUT"] = str(input_file)
        env["TAX_SIMULATOR_OUTPUT"] = str(output_dir)

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.tax_simulator_path),
                capture_output=True,
                text=True,
                timeout=120,
                env=env,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"Tax-Simulator failed with code {result.returncode}:\n"
                    f"stdout: {result.stdout}\n"
                    f"stderr: {result.stderr}"
                )

            # Parse output - look for results in expected locations
            return self._parse_output(output_dir, year)

        except subprocess.TimeoutExpired as e:
            raise RuntimeError("Tax-Simulator timed out after 120 seconds") from e

    def _parse_output(self, output_dir: Path, year: int) -> dict[str, float]:
        """Parse Tax-Simulator output files.

        The Tax-Simulator outputs results in several formats.
        We look for the detail files that contain per-unit calculations.
        """
        results = {}

        # Check for detail output
        detail_dir = output_dir / "detail"
        if detail_dir.exists():
            # Look for year-specific output
            for f in detail_dir.glob(f"*{year}*.csv"):
                results.update(self._parse_csv_output(f))

        # Also check for JSON summary
        summary_file = output_dir / "summary.json"
        if summary_file.exists():
            with open(summary_file) as f:
                summary = json.load(f)
                results.update(summary)

        return results

    def _parse_csv_output(self, csv_file: Path) -> dict[str, float]:
        """Parse a Tax-Simulator CSV output file."""
        import csv

        results = {}

        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Map common column names to our variables
                column_mapping = {
                    "income_tax": "income_tax",
                    "iitax": "income_tax",
                    "eitc": "eitc",
                    "ctc": "ctc",
                    "agi": "agi",
                    "c00100": "agi",
                    "standard": "standard_deduction",
                    "c04800": "taxable_income",
                    "taxable_income": "taxable_income",
                }

                for col, value in row.items():
                    col_lower = col.lower()
                    if col_lower in column_mapping:
                        with contextlib.suppress(ValueError, TypeError):
                            results[column_mapping[col_lower]] = float(value)

                # Only need first row for single-unit calculation
                break

        return results

    def validate(self, test_case: TestCase, variable: str, year: int = 2024) -> ValidatorResult:
        """Run validation using Yale Tax-Simulator.

        Note: This validator requires:
        1. R 4.0+ installed
        2. Yale Tax-Simulator repo cloned
        3. R packages installed (tidyverse, yaml, data.table, etc.)
        """
        var_lower = variable.lower()
        yale_var = VARIABLE_MAPPING.get(var_lower, var_lower)

        if var_lower not in SUPPORTED_VARIABLES:
            return ValidatorResult(
                validator_name=self.name,
                validator_type=self.validator_type,
                calculated_value=None,
                error=f"Variable '{variable}' not supported by Yale Tax-Simulator",
            )

        temp_dir = None
        try:
            # Create temporary directory for input/output
            temp_dir = Path(tempfile.mkdtemp(prefix="yale_tax_"))

            # Create input files
            input_file = self._create_tax_unit_input(test_case, year, temp_dir)
            runscript = self._create_runscript(year, temp_dir)

            # Run simulator
            results = self._run_simulator(input_file, runscript, temp_dir, year)

            # Extract requested variable
            calculated = results.get(yale_var)

            if calculated is None:
                return ValidatorResult(
                    validator_name=self.name,
                    validator_type=self.validator_type,
                    calculated_value=None,
                    error=f"Variable '{variable}' not found in Tax-Simulator output",
                    metadata={"available_vars": list(results.keys())},
                )

            return ValidatorResult(
                validator_name=self.name,
                validator_type=self.validator_type,
                calculated_value=calculated,
                metadata={
                    "yale_variable": yale_var,
                    "year": year,
                    "all_results": results,
                },
            )

        except FileNotFoundError as e:
            return ValidatorResult(
                validator_name=self.name,
                validator_type=self.validator_type,
                calculated_value=None,
                error=str(e),
            )
        except RuntimeError as e:
            return ValidatorResult(
                validator_name=self.name,
                validator_type=self.validator_type,
                calculated_value=None,
                error=f"Tax-Simulator execution failed: {e}",
            )
        except Exception as e:
            return ValidatorResult(
                validator_name=self.name,
                validator_type=self.validator_type,
                calculated_value=None,
                error=f"Unexpected error: {e}",
            )
        finally:
            # Cleanup temporary directory
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
