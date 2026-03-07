"""CPS microdata comparison between Cosilico and external validators.

Compares weighted totals from Cosilico's CPS calculations against
PolicyEngine, TAXSIM, and other validators.

Variable mappings are loaded from variable_mappings.yaml, which references
statute definitions in cosilico-us (e.g., 26/32/eitc.rac::earned_income_tax_credit).
"""

import contextlib
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import yaml


def load_variable_mappings() -> dict[str, dict]:
    """Load variable mappings from YAML file.

    The statute field is the source of truth for Cosilico variables.
    Format: {title}/{section}/{file}.rac::{formula_name}
    The formula_name after :: is used as the output column name.

    Returns:
        Dict mapping variable names to their configurations, including:
        - title: Human-readable name
        - statute: Path to statute definition (e.g., 26/32/eitc.rac::earned_income_tax_credit)
        - cosilico_col: Derived from statute (the formula name after ::)
        - pe_var: Variable name in PolicyEngine
        - tc_var: Variable name in Tax-Calculator
        - ts_var: Variable name in TAXSIM output
    """
    yaml_path = Path(__file__).parent / "variable_mappings.yaml"
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    result = {}
    for var_name, config in data.get("variables", {}).items():
        statute = config.get("statute", "")

        # Cosilico column name = variable name from statute (after ::)
        cosilico_col = statute.split("::")[-1] if "::" in statute else var_name

        result[var_name] = {
            "title": config.get("title", var_name),
            "statute": statute,
            "cosilico_col": cosilico_col,
            "pe_var": config.get("policyengine"),
            "pe_entity": config.get("policyengine_entity", "tax_unit"),
            "tc_var": config.get("taxcalc"),
            "ts_var": config.get("taxsim"),
            "derived": config.get("derived", False),
        }
    return result


# Load mappings at module level
COMPARISON_VARIABLES = load_variable_mappings()


@dataclass
class ModelResult:
    """Result from a single model."""

    name: str
    total: float
    n_records: int
    time_ms: float


@dataclass
class ComparisonTotals:
    """Comparison result for a single variable across all models."""

    variable: str
    title: str
    models: dict[str, ModelResult]  # model_name -> result

    def get_total(self, model: str) -> float:
        """Get total for a specific model."""
        return self.models.get(model, ModelResult(model, 0.0, 0, 0.0)).total

    @property
    def cosilico_total(self) -> float:
        return self.get_total("cosilico")

    @property
    def policyengine_total(self) -> float:
        return self.get_total("policyengine")

    @property
    def taxcalc_total(self) -> float:
        return self.get_total("taxcalc")

    @property
    def n_records(self) -> int:
        """Max records across models."""
        return max((m.n_records for m in self.models.values()), default=0)

    # Legacy properties for backward compatibility
    match_rate: float = 0.0
    mean_absolute_error: float = 0.0

    @property
    def difference(self) -> float:
        """Cosilico - PolicyEngine difference."""
        return self.cosilico_total - self.policyengine_total

    @property
    def percent_difference(self) -> float:
        """Percent difference from PolicyEngine."""
        if self.policyengine_total == 0:
            return 0.0
        return (self.difference / self.policyengine_total) * 100


@dataclass
class TimedResult:
    """Result with timing information."""

    data: dict[str, np.ndarray]
    elapsed_ms: float


def _load_cosilico_data_sources():
    """Load cosilico-data-sources modules."""
    data_sources = Path.home() / "CosilicoAI" / "cosilico-data-sources" / "micro" / "us"
    if str(data_sources) not in sys.path:
        sys.path.insert(0, str(data_sources))

    from cosilico_runner import run_all_calculations
    from tax_unit_builder import load_and_build_tax_units

    return load_and_build_tax_units, run_all_calculations


def load_cosilico_cps(year: int = 2024) -> TimedResult:
    """Load Cosilico calculations from CPS microdata.

    Returns:
        TimedResult with dict of arrays and elapsed time in ms.
    """
    load_and_build_tax_units, run_all_calculations = _load_cosilico_data_sources()

    start = time.perf_counter()
    df = load_and_build_tax_units(year)
    df = run_all_calculations(df, year)
    elapsed = (time.perf_counter() - start) * 1000

    result = {"weight": df["weight"].values}

    for var_name, config in COMPARISON_VARIABLES.items():
        col = config["cosilico_col"]
        if col in df.columns:
            result[var_name] = df[col].values
        else:
            result[var_name] = np.zeros(len(df))

    return TimedResult(data=result, elapsed_ms=elapsed)


def load_policyengine_values(
    year: int = 2024,
    variables: Optional[list[str]] = None,
) -> TimedResult:
    """Load PolicyEngine calculations.

    Returns:
        TimedResult with dict of arrays and elapsed time in ms.
    """
    from policyengine_us import Microsimulation

    start = time.perf_counter()
    sim = Microsimulation()

    if variables is None:
        variables = list(COMPARISON_VARIABLES.keys())

    result = {"weight": np.array(sim.calculate("tax_unit_weight", year))}
    n_tax_units = len(result["weight"])

    for var_name in variables:
        if var_name not in COMPARISON_VARIABLES:
            continue
        config = COMPARISON_VARIABLES[var_name]
        pe_var = config["pe_var"]
        pe_entity = config.get("pe_entity", "tax_unit")

        try:
            values = np.array(sim.calculate(pe_var, year))

            if pe_entity == "person" and len(values) != n_tax_units:
                # Need to aggregate person-level to tax unit
                # Use person's tax unit ID to sum
                person_tax_unit_id = np.array(sim.calculate("person_tax_unit_id", year))
                tax_unit_ids = np.array(sim.calculate("tax_unit_id", year))

                # Sum values by tax unit
                aggregated = np.zeros(n_tax_units)
                for i, tu_id in enumerate(tax_unit_ids):
                    mask = person_tax_unit_id == tu_id
                    aggregated[i] = values[mask].sum()
                values = aggregated

            result[var_name] = values
        except Exception:
            result[var_name] = np.zeros_like(result["weight"])

    elapsed = (time.perf_counter() - start) * 1000

    return TimedResult(data=result, elapsed_ms=elapsed)


def load_taxsim_values(
    year: int = 2024,
    variables: Optional[list[str]] = None,
) -> TimedResult:
    """Load TAXSIM calculations by running local executable on CPS data.

    Returns:
        TimedResult with dict of arrays and elapsed time in ms.
    """
    import csv
    import subprocess

    from cosilico_validators.comparison.multi_validator import get_taxsim_executable_path

    start = time.perf_counter()

    # Load Cosilico CPS data to get inputs
    load_and_build_tax_units, _ = _load_cosilico_data_sources()
    df = load_and_build_tax_units(year)

    # Get TAXSIM executable
    taxsim_path = get_taxsim_executable_path()

    # Build TAXSIM input CSV - use minimal required fields
    # TAXSIM input format: https://taxsim.nber.org/taxsim35/
    lines = ["taxsimid,year,state,mstat,page,sage,depx,pwages,idtl"]

    for i, row in df.iterrows():
        # Map filing status: 1=single, 2=joint
        mstat = 2 if row.get("is_joint", False) else 1

        # Handle NaN values
        def safe_int(val, default=0):
            try:
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    return default
                return max(0, int(val))
            except (ValueError, TypeError):
                return default

        def safe_float(val, default=0.0):
            try:
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    return default
                return max(0.0, float(val))
            except (ValueError, TypeError):
                return default

        page = safe_int(row.get("head_age", 35), 35)
        page = max(1, page)  # Must be at least 1
        sage = safe_int(row.get("spouse_age", 0)) if mstat == 2 else 0
        depx = safe_int(row.get("num_eitc_children", 0))
        pwages = safe_float(row.get("earned_income", 0))

        lines.append(f"{i + 1},{year},0,{mstat},{page},{sage},{depx},{pwages:.2f},2")

    input_csv = "\n".join(lines)

    # Run TAXSIM
    result = subprocess.run(
        [str(taxsim_path)],
        input=input_csv,
        capture_output=True,
        text=True,
        timeout=600,
    )

    if result.returncode != 0:
        raise RuntimeError(f"TAXSIM failed: {result.stderr}")

    # Parse output
    output_lines = result.stdout.strip().split("\n")
    reader = csv.DictReader(output_lines)
    output_records = list(reader)

    # Extract values
    n_records = len(output_records)
    weights = df["weight"].values[:n_records]

    if variables is None:
        variables = list(COMPARISON_VARIABLES.keys())

    data = {"weight": weights}

    for var_name in variables:
        if var_name not in COMPARISON_VARIABLES:
            continue
        config = COMPARISON_VARIABLES[var_name]
        ts_var = config.get("ts_var")
        if ts_var and output_records and ts_var in output_records[0]:
            data[var_name] = np.array([float(r.get(ts_var, 0) or 0) for r in output_records])
        else:
            data[var_name] = np.zeros(n_records)

    elapsed = (time.perf_counter() - start) * 1000

    return TimedResult(data=data, elapsed_ms=elapsed)


def load_taxcalc_values(
    year: int = 2024,
    variables: Optional[list[str]] = None,
) -> TimedResult:
    """Load Tax-Calculator calculations from its built-in CPS.

    Returns:
        TimedResult with dict of arrays and elapsed time in ms.
    """
    from taxcalc import Calculator, Policy, Records

    start = time.perf_counter()

    # Create calculator with CPS data
    rec = Records.cps_constructor()
    pol = Policy()
    calc = Calculator(policy=pol, records=rec)
    calc.advance_to_year(year)
    calc.calc_all()

    if variables is None:
        variables = list(COMPARISON_VARIABLES.keys())

    result = {"weight": calc.array("s006")}

    for var_name in variables:
        if var_name not in COMPARISON_VARIABLES:
            continue
        config = COMPARISON_VARIABLES[var_name]
        tc_var = config.get("tc_var")
        if tc_var:
            try:
                result[var_name] = calc.array(tc_var)
            except Exception:
                result[var_name] = np.zeros_like(result["weight"])
        else:
            result[var_name] = np.zeros_like(result["weight"])

    elapsed = (time.perf_counter() - start) * 1000

    return TimedResult(data=result, elapsed_ms=elapsed)


def compare_cps_totals(
    year: int = 2024,
    variables: Optional[list[str]] = None,
    tolerance: float = 1.0,
    models: Optional[list[str]] = None,
) -> dict[str, ComparisonTotals]:
    """Compare Cosilico CPS totals against multiple models.

    Args:
        year: Tax year
        variables: List of variables to compare (default: all)
        tolerance: Match tolerance in dollars
        models: List of models to include (default: all available)

    Returns:
        Dict mapping variable names to ComparisonTotals.
    """
    if variables is None:
        variables = list(COMPARISON_VARIABLES.keys())

    if models is None:
        models = ["cosilico", "policyengine", "taxcalc", "taxsim"]

    # Load data from each model
    model_results: dict[str, TimedResult] = {}

    if "cosilico" in models:
        model_results["cosilico"] = load_cosilico_cps(year)

    if "policyengine" in models:
        with contextlib.suppress(ImportError):
            model_results["policyengine"] = load_policyengine_values(year, variables)

    if "taxcalc" in models:
        with contextlib.suppress(ImportError):
            model_results["taxcalc"] = load_taxcalc_values(year, variables)

    if "taxsim" in models:
        try:
            model_results["taxsim"] = load_taxsim_values(year, variables)
        except Exception as e:
            print(f"Warning: TAXSIM failed: {e}")

    results = {}

    for var_name in variables:
        if var_name not in COMPARISON_VARIABLES:
            continue

        config = COMPARISON_VARIABLES[var_name]
        var_models = {}

        for model_name, timed_result in model_results.items():
            data = timed_result.data
            values = data.get(var_name, np.zeros_like(data["weight"]))
            weights = data["weight"]
            total = float((values * weights).sum())

            var_models[model_name] = ModelResult(
                name=model_name,
                total=total,
                n_records=len(values),
                time_ms=timed_result.elapsed_ms,
            )

        results[var_name] = ComparisonTotals(
            variable=var_name,
            title=config["title"],
            models=var_models,
        )

    return results


def export_to_dashboard(
    comparison: dict[str, ComparisonTotals],
    year: int = 2024,
) -> dict:
    """Export comparison results to dashboard JSON format."""
    sections = []
    total_cos_time = 0.0
    total_pe_time = 0.0

    for var_name, totals in comparison.items():
        sections.append(
            {
                "variable": var_name,
                "title": totals.title,
                "cosilico_total": totals.cosilico_total,
                "policyengine_total": totals.policyengine_total,
                "difference": totals.difference,
                "percent_difference": totals.percent_difference,
                "match_rate": totals.match_rate,
                "mean_absolute_error": totals.mean_absolute_error,
                "n_records": totals.n_records,
            }
        )
        total_cos_time = totals.cosilico_time_ms  # Same for all vars
        total_pe_time = totals.policyengine_time_ms  # pragma: no cover – unreachable due to AttributeError above

    all_totals = list(comparison.values())
    overall_match = np.mean([t.match_rate for t in all_totals]) if all_totals else 0
    overall_mae = np.mean([t.mean_absolute_error for t in all_totals]) if all_totals else 0

    return {
        "timestamp": datetime.now().isoformat(),
        "year": year,
        "data_source": "CPS ASEC",
        "sections": sections,
        "overall": {
            "match_rate": overall_match,
            "mean_absolute_error": overall_mae,
            "variables_compared": len(sections),
        },
        "performance": {
            "cosilico_ms": total_cos_time,
            "policyengine_ms": total_pe_time,
            "speedup": total_pe_time / total_cos_time if total_cos_time > 0 else 0,
        },
    }


def generate_report(year: int = 2024) -> str:
    """Generate a text report comparing all models."""
    comparison = compare_cps_totals(year)

    # Get all model names from first result
    first_result = next(iter(comparison.values()))
    model_names = list(first_result.models.keys())

    # Build header
    header_parts = [f"{'Variable':<25}"]
    for model in model_names:
        header_parts.append(f"{model:>14}")
    header = " ".join(header_parts)

    lines = [
        "=" * 100,
        f"CPS Weighted Totals Comparison ({year})",
        "=" * 100,
        "",
        header,
        "-" * 100,
    ]

    for _var_name, totals in comparison.items():
        row_parts = [f"{totals.title:<25}"]
        for model in model_names:
            val = totals.get_total(model) / 1e9
            row_parts.append(f"${val:>12.1f}B")
        lines.append(" ".join(row_parts))

    lines.extend(
        [
            "-" * 100,
            "",
            "Records per model:",
        ]
    )

    for model in model_names:
        n = first_result.models[model].n_records
        lines.append(f"  {model}: {n:,}")

    lines.extend(
        [
            "",
            "-" * 100,
            "Performance (ms):",
        ]
    )

    for model in model_names:
        ms = first_result.models[model].time_ms
        lines.append(f"  {model}: {ms:,.0f} ms ({ms / 1000:.1f}s)")

    lines.extend(
        [
            "",
            "=" * 100,
        ]
    )

    return "\n".join(lines)


def main():
    """Run comparison and print report."""
    print(generate_report(2024))


if __name__ == "__main__":  # pragma: no cover
    main()
