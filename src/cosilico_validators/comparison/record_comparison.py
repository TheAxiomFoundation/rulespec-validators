"""Record-by-record comparison across tax models.

All models run on the SAME input data from our CPS tax units.
This allows direct record-by-record comparison of outputs.
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class RecordComparison:
    """Comparison results for a single variable across all models."""

    variable: str
    n_records: int

    # Per-model arrays (same length, aligned by record)
    cosilico: np.ndarray
    policyengine: np.ndarray
    taxsim: np.ndarray
    taxcalc: np.ndarray
    weights: np.ndarray

    # Timing
    cosilico_ms: float
    policyengine_ms: float
    taxsim_ms: float
    taxcalc_ms: float

    @property
    def weighted_totals(self) -> dict[str, float]:
        return {
            "cosilico": float((self.cosilico * self.weights).sum()),
            "policyengine": float((self.policyengine * self.weights).sum()),
            "taxsim": float((self.taxsim * self.weights).sum()),
            "taxcalc": float((self.taxcalc * self.weights).sum()),
        }

    @property
    def mean_abs_diff_vs_pe(self) -> dict[str, float]:
        """Mean absolute difference vs PolicyEngine (weighted)."""
        pe = self.policyengine
        return {
            "cosilico": float((np.abs(self.cosilico - pe) * self.weights).sum() / self.weights.sum()),
            "taxsim": float((np.abs(self.taxsim - pe) * self.weights).sum() / self.weights.sum()),
            "taxcalc": float((np.abs(self.taxcalc - pe) * self.weights).sum() / self.weights.sum()),
        }

    @property
    def match_rate_vs_pe(self, tolerance: float = 1.0) -> dict[str, float]:
        """Fraction of records matching PolicyEngine within tolerance."""
        pe = self.policyengine
        w = self.weights
        total_weight = w.sum()
        return {
            "cosilico": float(((np.abs(self.cosilico - pe) <= tolerance) * w).sum() / total_weight),
            "taxsim": float(((np.abs(self.taxsim - pe) <= tolerance) * w).sum() / total_weight),
            "taxcalc": float(((np.abs(self.taxcalc - pe) <= tolerance) * w).sum() / total_weight),
        }


def load_cps_inputs(year: int = 2024) -> pd.DataFrame:
    """Load our CPS tax units - the common input for all models."""
    data_sources = Path.home() / "CosilicoAI" / "cosilico-data-sources" / "micro" / "us"
    if str(data_sources) not in sys.path:
        sys.path.insert(0, str(data_sources))

    from tax_unit_builder import load_and_build_tax_units

    return load_and_build_tax_units(year)


def run_cosilico(df: pd.DataFrame, year: int = 2024) -> tuple[pd.DataFrame, float]:
    """Run Cosilico on CPS data. Returns (results_df, elapsed_ms)."""
    data_sources = Path.home() / "CosilicoAI" / "cosilico-data-sources" / "micro" / "us"
    if str(data_sources) not in sys.path:
        sys.path.insert(0, str(data_sources))

    from cosilico_runner import run_all_calculations

    start = time.perf_counter()
    result = run_all_calculations(df.copy(), year)
    elapsed = (time.perf_counter() - start) * 1000

    return result, elapsed


def run_policyengine(df: pd.DataFrame, year: int = 2024) -> tuple[pd.DataFrame, float]:
    """Run PolicyEngine on CPS data. Returns (results_df, elapsed_ms)."""
    from policyengine_us import Simulation

    start = time.perf_counter()
    results = []

    # Run PE on each tax unit
    for _idx, row in df.iterrows():
        situation = _create_pe_situation(row, year)
        sim = Simulation(situation=situation)

        result = {
            "eitc": float(sim.calculate("eitc", year).sum()),
            "non_refundable_ctc": float(sim.calculate("non_refundable_ctc", year).sum()),
            "refundable_ctc": float(sim.calculate("refundable_ctc", year).sum()),
            "income_tax_before_credits": float(sim.calculate("income_tax_before_credits", year).sum()),
            "self_employment_tax": float(sim.calculate("self_employment_tax", year).sum()),
            "adjusted_gross_income": float(sim.calculate("adjusted_gross_income", year).sum()),
        }
        results.append(result)

    elapsed = (time.perf_counter() - start) * 1000
    result_df = pd.DataFrame(results, index=df.index)

    return result_df, elapsed


def _safe_int(val, default: int = 0) -> int:
    """Safely convert to int, handling NaN."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    return int(val)


def _safe_float(val, default: float = 0.0) -> float:
    """Safely convert to float, handling NaN."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    return float(val)


def _create_pe_situation(row: pd.Series, year: int) -> dict:
    """Create PolicyEngine situation from a CPS tax unit row."""
    is_joint = bool(row.get("is_joint", False))
    n_deps = _safe_int(row.get("num_dependents", 0))

    # Build members list
    members = ["head"]
    if is_joint:
        members.append("spouse")
    for i in range(n_deps):
        members.append(f"dep{i + 1}")

    # People
    people = {
        "head": {
            "age": {str(year): _safe_int(row.get("head_age"), 40)},
            "employment_income": {str(year): _safe_float(row.get("wage_income"))},
            "self_employment_income": {str(year): _safe_float(row.get("self_employment_income"))},
            "social_security": {str(year): _safe_float(row.get("social_security_income"))},
            "taxable_interest_income": {str(year): _safe_float(row.get("interest_income"))},
            "qualified_dividend_income": {str(year): _safe_float(row.get("dividend_income"))},
            "rental_income": {str(year): _safe_float(row.get("rental_income"))},
            "unemployment_compensation": {str(year): _safe_float(row.get("unemployment_compensation"))},
        }
    }

    if is_joint:
        spouse_age = _safe_int(row.get("spouse_age"), 40)
        people["spouse"] = {
            "age": {str(year): spouse_age if spouse_age > 0 else 40},
        }

    # Dependents - use CPS counts for EITC/CTC qualifying children
    n_eitc_children = _safe_int(row.get("num_eitc_children"))
    n_ctc_children = _safe_int(row.get("num_ctc_children"))
    _safe_int(row.get("num_other_dependents", 0))

    # EITC-qualifying children (under 19, or under 24 if student)
    for i in range(n_eitc_children):
        dep_name = f"dep{i + 1}"
        people[dep_name] = {
            "age": {str(year): 10},  # Young child - qualifies for EITC
            "is_tax_unit_dependent": {str(year): True},
        }

    # CTC-only children (under 17 but possibly not EITC qualifying)
    # These are rare - usually EITC children >= CTC children
    ctc_only = max(0, n_ctc_children - n_eitc_children)
    for i in range(ctc_only):
        dep_name = f"dep{n_eitc_children + i + 1}"
        people[dep_name] = {
            "age": {str(year): 10},  # Young child for CTC
            "is_tax_unit_dependent": {str(year): True},
        }

    # Other dependents (19+, don't qualify for EITC/CTC child credit)
    # These get the $500 other dependent credit
    other_deps_count = n_deps - n_eitc_children - ctc_only
    for i in range(max(0, other_deps_count)):
        dep_name = f"dep{n_eitc_children + ctc_only + i + 1}"
        people[dep_name] = {
            "age": {str(year): 20},  # Adult dependent - no EITC, $500 ODC only
            "is_tax_unit_dependent": {str(year): True},
        }

    situation = {
        "people": people,
        "tax_units": {
            "tax_unit": {
                "members": members,
            }
        },
        "families": {
            "family": {
                "members": members,
            }
        },
        "spm_units": {
            "spm_unit": {
                "members": members,
            }
        },
        "households": {
            "household": {
                "members": members,
                "state_name": {str(year): "CA"},  # Default state
            }
        },
    }

    return situation


def run_taxsim(df: pd.DataFrame, year: int = 2024) -> tuple[pd.DataFrame, float]:
    """Run TAXSIM on CPS data. Returns (results_df, elapsed_ms)."""
    import csv
    import subprocess

    from cosilico_validators.comparison.multi_validator import get_taxsim_executable_path

    start = time.perf_counter()

    taxsim_path = get_taxsim_executable_path()

    # Build TAXSIM input
    lines = [
        "taxsimid,year,state,mstat,page,sage,depx,pwages,swages,dividends,intrec,ltcg,stcg,otherprop,pensions,gssi,psemp,ssemp,idtl"
    ]

    for i, (_idx, row) in enumerate(df.iterrows()):
        mstat = 2 if row.get("is_joint", False) else 1
        page = max(1, _safe_int(row.get("head_age"), 40))
        sage = _safe_int(row.get("spouse_age"), 0) if mstat == 2 else 0
        depx = _safe_int(row.get("num_dependents"), 0)

        pwages = max(0, _safe_float(row.get("wage_income")))
        swages = 0  # We don't split wages
        dividends = max(0, _safe_float(row.get("dividend_income")))
        intrec = max(0, _safe_float(row.get("interest_income")))
        ltcg = 0
        stcg = 0
        otherprop = max(0, _safe_float(row.get("rental_income")))
        pensions = 0
        gssi = max(0, _safe_float(row.get("social_security_income")))
        psemp = max(0, _safe_float(row.get("self_employment_income")))
        ssemp = 0

        lines.append(
            f"{i + 1},{year},0,{mstat},{page},{sage},{depx},{pwages:.2f},{swages:.2f},{dividends:.2f},{intrec:.2f},{ltcg:.2f},{stcg:.2f},{otherprop:.2f},{pensions:.2f},{gssi:.2f},{psemp:.2f},{ssemp:.2f},2"
        )

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

    # Map TAXSIM output to our variables
    results = []
    for rec in output_records:
        results.append(
            {
                "eitc": float(rec.get("v25", 0) or 0),
                "non_refundable_ctc": float(rec.get("v22", 0) or 0),
                "refundable_ctc": float(rec.get("actc", 0) or 0),
                "income_tax_before_credits": float(rec.get("v19", 0) or 0),
                "adjusted_gross_income": float(rec.get("v10", 0) or 0),
            }
        )

    elapsed = (time.perf_counter() - start) * 1000
    result_df = pd.DataFrame(results, index=df.index[: len(results)])

    return result_df, elapsed


def compare_records(
    year: int = 2024,
    variables: Optional[list[str]] = None,
    sample_size: Optional[int] = None,
) -> dict[str, RecordComparison]:
    """Run all models on same CPS data and compare record by record.

    Args:
        year: Tax year
        variables: Variables to compare (default: core set)
        sample_size: Limit to N records for faster testing

    Returns:
        Dict mapping variable names to RecordComparison objects
    """
    if variables is None:
        variables = ["eitc", "non_refundable_ctc", "refundable_ctc"]

    # Load common input data
    print("Loading CPS inputs...")
    df = load_cps_inputs(year)

    if sample_size:
        df = df.head(sample_size)

    print(f"Running on {len(df):,} tax units...")

    # Run each model on same data
    print("  Running Cosilico...")
    cos_df, cos_ms = run_cosilico(df, year)

    print("  Running PolicyEngine...")
    pe_df, pe_ms = run_policyengine(df, year)

    print("  Running TAXSIM...")
    ts_df, ts_ms = run_taxsim(df, year)

    # Build comparison for each variable
    results = {}
    weights = df["weight"].values

    for var in variables:
        cos_col = var
        if var == "non_refundable_ctc":
            cos_col = "non_refundable_ctc"
        elif var == "refundable_ctc":
            cos_col = "refundable_ctc"

        results[var] = RecordComparison(
            variable=var,
            n_records=len(df),
            cosilico=cos_df[cos_col].values if cos_col in cos_df.columns else np.zeros(len(df)),
            policyengine=pe_df[var].values if var in pe_df.columns else np.zeros(len(df)),
            taxsim=ts_df[var].values if var in ts_df.columns else np.zeros(len(df)),
            taxcalc=np.zeros(len(df)),  # TODO: add Tax-Calculator
            weights=weights,
            cosilico_ms=cos_ms,
            policyengine_ms=pe_ms,
            taxsim_ms=ts_ms,
            taxcalc_ms=0,
        )

    return results


def print_comparison(results: dict[str, RecordComparison]):
    """Print comparison results."""
    print("\n" + "=" * 80)
    print("RECORD-BY-RECORD COMPARISON (same input data)")
    print("=" * 80)

    for var, comp in results.items():
        totals = comp.weighted_totals
        print(f"\n{var.upper()}")
        print("-" * 40)
        print(f"  Records: {comp.n_records:,}")
        print("  Weighted totals:")
        for model, total in totals.items():
            print(f"    {model:15} ${total / 1e9:>10.1f}B")

        mae = comp.mean_abs_diff_vs_pe
        print("  Mean abs diff vs PE:")
        for model, diff in mae.items():
            print(f"    {model:15} ${diff:>10.0f}")


if __name__ == "__main__":  # pragma: no cover
    # Quick test with small sample
    results = compare_records(year=2024, sample_size=100)
    print_comparison(results)
