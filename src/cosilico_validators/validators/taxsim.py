"""TAXSIM validator - calls NBER TAXSIM via web API or local executable.

TAXSIM is the Tax Simulation Model from the National Bureau of Economic Research.
It provides authoritative tax calculations for federal and state income taxes.

Two execution modes are supported:
1. Web API (default): Uses TAXSIM's HTTP service at taxsim.nber.org
2. Local executable: Uses a downloaded TAXSIM binary (like policyengine-taxsim)

Web API documentation: https://taxsim.nber.org/taxsim35/low-level-remote.html
Variable documentation: https://taxsim.nber.org/taxsim35/
"""

import csv
import io
import os
import platform
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Literal

from cosilico_validators.validators.base import (
    BaseValidator,
    TestCase,
    ValidatorResult,
    ValidatorType,
)

# TAXSIM output variable column names (for idtl=2 full output)
# See: https://taxsim.nber.org/taxsim35/
# Full variable list: https://cran.rstudio.com/web/packages/usincometaxes/vignettes/taxsim-output.html
TAXSIM_OUTPUT_VARS = {
    # Primary tax outputs
    "federal_income_tax": "fiitax",
    "fiitax": "fiitax",
    "income_tax": "fiitax",  # Alias for federal income tax
    "state_income_tax": "siitax",
    "siitax": "siitax",
    "fica": "fica",
    # AGI and taxable income
    "agi": "v10",
    "adjusted_gross_income": "v10",
    "federal_agi": "v10",
    "taxable_income": "v18",
    "federal_taxable_income": "v18",
    # Deductions
    "standard_deduction": "v13",  # Zero bracket amount (standard deduction equivalent)
    "zero_bracket_amount": "v13",
    "deductions": "v16",
    "itemized_deductions": "v17",
    # Credits
    "ctc": "v22",
    "child_tax_credit": "v22",
    "actc": "v23",
    "refundable_ctc": "v23",
    "additional_child_tax_credit": "v23",
    "cdctc": "v24",
    "child_care_credit": "v24",
    "eitc": "v25",
    "earned_income_credit": "v25",
    # AMT
    "amt": "v27",
    "alternative_minimum_tax": "v27",
    "amt_income": "v26",
    # State outputs
    "state_agi": "v32",
    "state_standard_deduction": "v34",
    "state_taxable_income": "v36",
    "state_eitc": "v39",
    # Other
    "exemptions": "v14",
    "exemption_phaseout": "v15",
    "tax_before_credits": "v19",
}

# State FIPS codes
STATE_CODES = {
    "AL": 1,
    "AK": 2,
    "AZ": 4,
    "AR": 5,
    "CA": 6,
    "CO": 8,
    "CT": 9,
    "DE": 10,
    "DC": 11,
    "FL": 12,
    "GA": 13,
    "HI": 15,
    "ID": 16,
    "IL": 17,
    "IN": 18,
    "IA": 19,
    "KS": 20,
    "KY": 21,
    "LA": 22,
    "ME": 23,
    "MD": 24,
    "MA": 25,
    "MI": 26,
    "MN": 27,
    "MS": 28,
    "MO": 29,
    "MT": 30,
    "NE": 31,
    "NV": 32,
    "NH": 33,
    "NJ": 34,
    "NM": 35,
    "NY": 36,
    "NC": 37,
    "ND": 38,
    "OH": 39,
    "OK": 40,
    "OR": 41,
    "PA": 42,
    "RI": 44,
    "SC": 45,
    "SD": 46,
    "TN": 47,
    "TX": 48,
    "UT": 49,
    "VT": 50,
    "VA": 51,
    "WA": 53,
    "WV": 54,
    "WI": 55,
    "WY": 56,
}

# Filing status mapping
# See: https://taxsim.nber.org/taxsim35/
# 1 = single, 2 = married filing jointly, 3 = head of household,
# 6 = married filing separately, 8 = dependent taxpayer
MSTAT_CODES = {
    "SINGLE": 1,
    "JOINT": 2,
    "MARRIED_FILING_JOINTLY": 2,
    "HEAD_OF_HOUSEHOLD": 3,
    "HOH": 3,
    "MARRIED_FILING_SEPARATELY": 6,
    "SEPARATE": 6,
    "MFS": 6,
    "DEPENDENT": 8,
}

# TAXSIM web API endpoint
TAXSIM_API_URL = "https://taxsim.nber.org/taxsim35/redirect.cgi"

# TAXSIM input columns in order
TAXSIM_COLUMNS = [
    "taxsimid",
    "year",
    "state",
    "mstat",
    "page",
    "sage",
    "depx",
    "age1",
    "age2",
    "age3",
    "pwages",
    "swages",
    "psemp",
    "ssemp",
    "dividends",
    "intrec",
    "stcg",
    "ltcg",
    "otherprop",
    "nonprop",
    "pensions",
    "gssi",
    "pui",
    "sui",
    "transfers",
    "rentpaid",
    "proptax",
    "otheritem",
    "childcare",
    "mortgage",
    "scorp",
    "pbusinc",
    "pprofinc",
    "sbusinc",
    "sprofinc",
    "idtl",
]


class TaxsimValidator(BaseValidator):
    """Validator using NBER TAXSIM via web API or local executable.

    TAXSIM (Tax Simulation Model) is maintained by NBER and provides
    authoritative federal and state income tax calculations.

    Two execution modes are supported:
    - "web" (default): Uses the TAXSIM HTTP API at taxsim.nber.org
    - "local": Uses a downloaded TAXSIM executable (requires download)

    Web API is recommended as it requires no setup and stays current.

    Example:
        >>> validator = TaxsimValidator()  # Uses web API
        >>> result = validator.validate(test_case, "eitc", year=2023)
        >>> print(result.calculated_value)

    Note: TAXSIM-35 supports tax years 1960-2023.
    """

    name = "TAXSIM"
    validator_type = ValidatorType.REFERENCE
    supported_variables = set(TAXSIM_OUTPUT_VARS.keys())

    def __init__(
        self,
        mode: Literal["web", "local"] = "web",
        taxsim_path: str | Path | None = None,
        max_retries: int = 3,
        timeout: int = 60,
    ):
        """Initialize TAXSIM validator.

        Args:
            mode: Execution mode - "web" (default) or "local"
            taxsim_path: Path to TAXSIM executable (only for local mode).
                         If not provided, searches standard locations.
            max_retries: Number of retry attempts for web API (default: 3)
            timeout: Request timeout in seconds for web API (default: 60)
        """
        self.mode = mode
        self.max_retries = max_retries
        self.timeout = timeout

        if mode == "local":
            self.taxsim_path = self._resolve_taxsim_path(taxsim_path)
        else:
            self.taxsim_path = None

    def _resolve_taxsim_path(self, provided_path: str | Path | None) -> Path:
        """Find the TAXSIM executable."""
        if provided_path:
            path = Path(provided_path)
            if path.exists():
                return path
            raise FileNotFoundError(f"TAXSIM executable not found at: {path}")

        # Detect OS-specific executable name
        system = platform.system().lower()
        if system == "darwin":
            exe_name = "taxsim35-osx.exe"
        elif system == "windows":
            exe_name = "taxsim35-windows.exe"
        elif system == "linux":
            exe_name = "taxsim35-unix.exe"
        else:
            raise OSError(f"Unsupported operating system: {system}")

        # Search paths
        search_paths = [
            Path(__file__).parent.parent.parent.parent / "resources" / "taxsim" / exe_name,
            Path.cwd() / "resources" / "taxsim" / exe_name,
            Path.home() / ".cosilico" / "taxsim" / exe_name,
        ]

        for path in search_paths:
            if path.exists():
                return path

        raise FileNotFoundError(
            f"TAXSIM executable '{exe_name}' not found. "
            f"Download from https://taxsim.nber.org/taxsim35/ and place in one of:\n"
            + "\n".join(f"  - {p}" for p in search_paths)
        )

    def supports_variable(self, variable: str) -> bool:
        return variable.lower() in TAXSIM_OUTPUT_VARS

    def _build_taxsim_input(self, test_case: TestCase, year: int) -> dict[str, Any]:
        """Convert test case to TAXSIM input format."""
        inputs = test_case.inputs

        # Default TAXSIM record
        taxsim_input = {
            "taxsimid": 1,
            "year": year,
            "state": 6,  # California default
            "mstat": 1,  # Single
            "page": 30,  # Primary age
            "sage": 0,  # Spouse age
            "depx": 0,  # Number of dependents
            "pwages": 0,  # Primary wages
            "swages": 0,  # Spouse wages
            "idtl": 2,  # Full output
        }

        # Map common inputs to TAXSIM variables
        input_mapping = {
            "age": "page",
            "age_at_end_of_year": "page",
            "earned_income": "pwages",
            "employment_income": "pwages",
            "wages": "pwages",
            "spouse_wages": "swages",
            "qualifying_children": "depx",
            "eitc_qualifying_children_count": "depx",
            "num_children": "depx",
            "children": "depx",
        }

        for key, value in inputs.items():
            key_lower = key.lower()

            # Handle state
            if key_lower in ["state", "state_name"]:
                if isinstance(value, str):
                    taxsim_input["state"] = STATE_CODES.get(value.upper(), 6)
                else:
                    taxsim_input["state"] = value
            # Handle filing status
            elif key_lower == "filing_status":
                taxsim_input["mstat"] = MSTAT_CODES.get(value.upper(), 1)
            # Handle mapped inputs
            elif key_lower in input_mapping:
                taxsim_input[input_mapping[key_lower]] = value

        # Handle filing status affecting household structure
        filing_status = inputs.get("filing_status", "SINGLE").upper()
        if filing_status in ["JOINT", "MARRIED_FILING_JOINTLY"]:
            taxsim_input["mstat"] = 2
            if taxsim_input["sage"] == 0:
                taxsim_input["sage"] = taxsim_input["page"]  # Default spouse age

        # Add child ages if dependents exist
        num_deps = taxsim_input.get("depx", 0)
        for i in range(min(num_deps, 3)):
            taxsim_input[f"age{i + 1}"] = 10  # Default child age

        return taxsim_input

    def _create_input_csv(self, taxsim_input: dict) -> str:
        """Create TAXSIM input CSV file."""
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)

        # Write header
        temp_file.write(",".join(TAXSIM_COLUMNS) + "\n")

        # Write data row
        row = [str(taxsim_input.get(col, 0)) for col in TAXSIM_COLUMNS]
        temp_file.write(",".join(row) + "\n")

        temp_file.close()
        return temp_file.name

    def _create_csv_string(self, taxsim_input: dict) -> str:
        """Create TAXSIM input as a CSV string (for web API)."""
        output = io.StringIO()
        writer = csv.writer(output)

        # Write header
        writer.writerow(TAXSIM_COLUMNS)

        # Write data row
        row = [taxsim_input.get(col, 0) for col in TAXSIM_COLUMNS]
        writer.writerow(row)

        return output.getvalue()

    def _execute_web(self, csv_data: str) -> str:
        """Execute TAXSIM via web API and return output.

        Uses curl for multipart form upload per TAXSIM documentation.
        See: https://taxsim.nber.org/taxsim35/low-level-remote.html
        """
        for attempt in range(self.max_retries):
            try:
                # Write CSV to temp file for curl
                with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
                    f.write(csv_data)
                    temp_path = f.name

                try:
                    # Use curl for multipart form upload
                    result = subprocess.run(
                        [
                            "curl",
                            "-s",  # Silent
                            "-F",
                            f"txpydata.csv=@{temp_path}",
                            TAXSIM_API_URL,
                        ],
                        capture_output=True,
                        text=True,
                        timeout=self.timeout,
                    )

                    if result.returncode != 0:
                        raise RuntimeError(f"curl error: {result.stderr}")

                    response = result.stdout

                    # Check for error responses
                    if not response.strip():
                        raise RuntimeError("Empty response from TAXSIM API")

                    if "<html" in response.lower() or "error" in response.lower()[:100]:
                        raise RuntimeError(f"TAXSIM API error: {response[:200]}")

                    return response

                finally:
                    Path(temp_path).unlink(missing_ok=True)

            except subprocess.TimeoutExpired as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)  # Exponential backoff
                    continue
                raise RuntimeError(f"TAXSIM API timeout after {self.timeout}s") from e

            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)
                    continue
                raise RuntimeError(f"TAXSIM API failed: {e}") from e

        raise RuntimeError("TAXSIM API failed after all retries")  # pragma: no cover – loop always returns/raises

    def _execute_local(self, input_file: str) -> str:
        """Execute TAXSIM locally and return output."""
        if self.taxsim_path is None:
            raise RuntimeError("Local mode requires TAXSIM executable path")

        # Make executable on Unix
        if platform.system().lower() != "windows":
            os.chmod(self.taxsim_path, 0o755)

        # Create output file
        output_fd, output_file = tempfile.mkstemp(suffix=".csv")
        os.close(output_fd)

        try:
            system = platform.system().lower()

            if system != "windows":
                cmd = f'cat "{input_file}" | "{self.taxsim_path}" > "{output_file}"'
            else:  # pragma: no cover – Windows-only path
                cmd = f'type "{input_file}" | "{self.taxsim_path}" > "{output_file}"'

            # Set up environment
            env = os.environ.copy()
            if system == "darwin":
                homebrew_paths = ["/opt/homebrew/bin", "/usr/local/bin"]
                current_path = env.get("PATH", "")
                for hb_path in reversed(homebrew_paths):
                    if hb_path not in current_path:
                        current_path = f"{hb_path}:{current_path}"
                env["PATH"] = current_path

            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                env=env,
            )

            if result.returncode != 0:
                raise RuntimeError(f"TAXSIM failed: {result.stderr}")

            with open(output_file, "r") as f:
                return f.read()

        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)

    def _parse_output(self, output: str, variable: str) -> float | None:
        """Parse TAXSIM output CSV."""
        lines = output.strip().split("\n")
        if len(lines) < 2:
            raise ValueError(f"Invalid TAXSIM output: {output}")

        headers = [h.strip() for h in lines[0].split(",")]
        values = [v.strip() for v in lines[1].split(",")]

        result = dict(zip(headers, values))

        # Get the column name for this variable
        var_lower = variable.lower()
        col_name = TAXSIM_OUTPUT_VARS.get(var_lower)

        if col_name and col_name in result:
            return float(result[col_name])

        # Try direct lookup
        if var_lower in result:
            return float(result[var_lower])

        # Try partial match
        for key, value in result.items():
            if var_lower in key.lower():
                return float(value)

        return None

    def validate(self, test_case: TestCase, variable: str, year: int = 2023) -> ValidatorResult:
        """Run validation using TAXSIM (web API or local executable).

        Args:
            test_case: Test case with inputs
            variable: Variable to calculate (e.g., "eitc", "agi", "income_tax")
            year: Tax year (1960-2023 supported by TAXSIM-35)

        Returns:
            ValidatorResult with calculated value or error

        Note: TAXSIM-35 only supports tax years 1960-2023.
        """
        # Validate year is within TAXSIM's supported range
        if year < 1960 or year > 2023:
            return ValidatorResult(
                validator_name=self.name,
                validator_type=self.validator_type,
                calculated_value=None,
                error=f"TAXSIM-35 only supports tax years 1960-2023, got {year}",
            )

        var_lower = variable.lower()
        if var_lower not in TAXSIM_OUTPUT_VARS:
            return ValidatorResult(
                validator_name=self.name,
                validator_type=self.validator_type,
                calculated_value=None,
                error=f"Variable '{variable}' not supported by TAXSIM",
            )

        input_file = None
        try:
            taxsim_input = self._build_taxsim_input(test_case, year)

            # Execute based on mode
            if self.mode == "web":
                csv_data = self._create_csv_string(taxsim_input)
                output = self._execute_web(csv_data)
            else:  # local mode
                input_file = self._create_input_csv(taxsim_input)
                output = self._execute_local(input_file)

            calculated = self._parse_output(output, variable)

            if calculated is None:
                return ValidatorResult(
                    validator_name=self.name,
                    validator_type=self.validator_type,
                    calculated_value=None,
                    error=f"Could not find {variable} in TAXSIM output",
                    metadata={"raw_output": output, "mode": self.mode},
                )

            return ValidatorResult(
                validator_name=self.name,
                validator_type=self.validator_type,
                calculated_value=calculated,
                metadata={"taxsim_input": taxsim_input, "year": year, "mode": self.mode},
            )

        except FileNotFoundError as e:
            return ValidatorResult(
                validator_name=self.name,
                validator_type=self.validator_type,
                calculated_value=None,
                error=str(e),
            )
        except Exception as e:
            return ValidatorResult(
                validator_name=self.name,
                validator_type=self.validator_type,
                calculated_value=None,
                error=f"TAXSIM execution failed: {e}",
            )
        finally:
            if input_file and os.path.exists(input_file):
                os.unlink(input_file)

    def batch_validate(self, test_cases: list[TestCase], variable: str, year: int = 2023) -> list[ValidatorResult]:
        """Validate multiple test cases efficiently.

        For web API mode, batches requests to minimize API calls.
        TAXSIM can handle up to ~2000 records per request.

        Args:
            test_cases: List of test cases to validate
            variable: Variable to calculate
            year: Tax year

        Returns:
            List of ValidatorResult for each test case
        """
        if not test_cases:
            return []

        # Validate inputs
        if year < 1960 or year > 2023:
            return [
                ValidatorResult(
                    validator_name=self.name,
                    validator_type=self.validator_type,
                    calculated_value=None,
                    error=f"TAXSIM-35 only supports tax years 1960-2023, got {year}",
                )
                for _ in test_cases
            ]

        var_lower = variable.lower()
        if var_lower not in TAXSIM_OUTPUT_VARS:
            return [
                ValidatorResult(
                    validator_name=self.name,
                    validator_type=self.validator_type,
                    calculated_value=None,
                    error=f"Variable '{variable}' not supported by TAXSIM",
                )
                for _ in test_cases
            ]

        # For local mode, use sequential execution
        if self.mode == "local":
            return super().batch_validate(test_cases, variable, year)

        # For web mode, batch the request
        try:
            # Build all inputs
            taxsim_inputs = []
            for i, tc in enumerate(test_cases, start=1):
                ti = self._build_taxsim_input(tc, year)
                ti["taxsimid"] = i  # Use index as ID
                taxsim_inputs.append(ti)

            # Create combined CSV
            output_buf = io.StringIO()
            writer = csv.writer(output_buf)
            writer.writerow(TAXSIM_COLUMNS)
            for ti in taxsim_inputs:
                row = [ti.get(col, 0) for col in TAXSIM_COLUMNS]
                writer.writerow(row)

            csv_data = output_buf.getvalue()
            output = self._execute_web(csv_data)

            # Parse batch output
            lines = output.strip().split("\n")
            if len(lines) < 2:
                raise ValueError(f"Invalid TAXSIM batch output: {output[:200]}")

            headers = [h.strip() for h in lines[0].split(",")]
            results_by_id = {}

            for line in lines[1:]:
                if not line.strip():
                    continue
                values = [v.strip() for v in line.split(",")]
                row_dict = dict(zip(headers, values))
                taxsim_id = int(float(row_dict.get("taxsimid", 0)))
                results_by_id[taxsim_id] = row_dict

            # Map results back to test cases
            results = []
            col_name = TAXSIM_OUTPUT_VARS.get(var_lower)

            for i, _tc in enumerate(test_cases, start=1):
                row_dict = results_by_id.get(i)
                if row_dict is None:
                    results.append(
                        ValidatorResult(
                            validator_name=self.name,
                            validator_type=self.validator_type,
                            calculated_value=None,
                            error=f"No TAXSIM result for case {i}",
                        )
                    )
                    continue

                value = None
                if col_name and col_name in row_dict:
                    value = float(row_dict[col_name])
                elif var_lower in row_dict:
                    value = float(row_dict[var_lower])

                if value is None:
                    results.append(
                        ValidatorResult(
                            validator_name=self.name,
                            validator_type=self.validator_type,
                            calculated_value=None,
                            error=f"Could not find {variable} in TAXSIM output",
                            metadata={"mode": self.mode},
                        )
                    )
                else:
                    results.append(
                        ValidatorResult(
                            validator_name=self.name,
                            validator_type=self.validator_type,
                            calculated_value=value,
                            metadata={
                                "taxsim_input": taxsim_inputs[i - 1],
                                "year": year,
                                "mode": self.mode,
                            },
                        )
                    )

            return results

        except Exception as e:
            # If batch fails, return error for all cases
            return [
                ValidatorResult(
                    validator_name=self.name,
                    validator_type=self.validator_type,
                    calculated_value=None,
                    error=f"TAXSIM batch execution failed: {e}",
                )
                for _ in test_cases
            ]
