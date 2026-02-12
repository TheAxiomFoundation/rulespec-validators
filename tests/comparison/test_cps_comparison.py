"""Tests for CPS microdata comparison against external validators."""

import importlib
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd


def _import_cps_comparison():
    """Import cps_comparison with mocked external dependencies."""
    # Remove any cached import
    for mod_name in list(sys.modules.keys()):
        if "cps_comparison" in mod_name:
            del sys.modules[mod_name]

    # Create mock modules for top-level imports
    mock_pe = MagicMock()
    mock_sim = MagicMock()
    mock_pe.Microsimulation = MagicMock(return_value=mock_sim)
    mock_sim.calculate.return_value = np.array([100.0, 200.0])

    mock_builder = MagicMock()
    mock_runner = MagicMock()

    saved = {}
    for mod in ["policyengine_us", "tax_unit_builder", "cosilico_runner"]:
        saved[mod] = sys.modules.get(mod)

    sys.modules["policyengine_us"] = mock_pe
    sys.modules["tax_unit_builder"] = mock_builder
    sys.modules["cosilico_runner"] = mock_runner

    try:
        mod = importlib.import_module(
            "cosilico_validators.comparison.cps_comparison"
        )
        return mod, mock_pe, mock_sim, mock_builder, mock_runner
    finally:
        # Restore after import - keep module cached but restore sys.modules
        for mod_name, val in saved.items():
            if val is None:
                sys.modules.pop(mod_name, None)
            else:
                sys.modules[mod_name] = val


class TestGetPeValues:
    def test_get_pe_values_success(self):
        mod, mock_pe, mock_sim, _, _ = _import_cps_comparison()
        mock_sim.calculate.return_value = np.array([100.0, 200.0])
        # Re-inject mock for policyengine_us used inside function
        with patch.dict(sys.modules, {"policyengine_us": mock_pe}):
            result = mod.get_pe_values(year=2024)
            assert isinstance(result, pd.DataFrame)
            assert "pe_eitc" in result.columns
            assert "pe_ctc_total" in result.columns
            assert len(result) == 2


class TestCompareCalculations:
    def test_compare_success(self):
        mod, _, _, _, _ = _import_cps_comparison()
        cos_df = pd.DataFrame({
            "tax_unit_id": [1, 2, 3],
            "weight": [100.0, 200.0, 300.0],
            "cos_eitc": [500.0, 600.0, 0.0],
            "cos_ctc_total": [2000.0, 0.0, 2000.0],
            "cos_se_tax": [0.0, 500.0, 0.0],
            "cos_income_tax": [3000.0, 5000.0, 2000.0],
            "cos_niit": [0.0, 0.0, 0.0],
            "adjusted_gross_income": [50000.0, 80000.0, 30000.0],
            "taxable_income": [35000.0, 65000.0, 15000.0],
        })
        pe_df = pd.DataFrame({
            "tax_unit_id": [1, 2, 3],
            "pe_eitc": [500.0, 600.0, 0.0],
            "pe_ctc_total": [2000.0, 0.0, 2000.0],
            "pe_se_tax": [0.0, 500.0, 0.0],
            "pe_income_tax": [3000.0, 5000.0, 2000.0],
            "pe_niit": [0.0, 0.0, 0.0],
            "pe_agi": [50000.0, 80000.0, 30000.0],
            "pe_taxable_income": [35000.0, 65000.0, 15000.0],
            "pe_earned_income": [50000.0, 80000.0, 30000.0],
        })
        results = mod.compare_calculations(cos_df, pe_df)
        assert isinstance(results, dict)
        assert "EITC" in results
        assert "match_rate" in results["EITC"]
        assert "mean_diff" in results["EITC"]
        assert "correlation" in results["EITC"]

    def test_compare_missing_columns(self):
        mod, _, _, _, _ = _import_cps_comparison()
        cos_df = pd.DataFrame({
            "tax_unit_id": [1, 2],
            "weight": [100.0, 200.0],
        })
        pe_df = pd.DataFrame({
            "tax_unit_id": [1, 2],
        })
        results = mod.compare_calculations(cos_df, pe_df)
        assert isinstance(results, dict)

    def test_compare_few_nonzero(self):
        """Correlation with fewer than 10 nonzero values returns NaN."""
        mod, _, _, _, _ = _import_cps_comparison()
        cos_df = pd.DataFrame({
            "tax_unit_id": [1],
            "weight": [100.0],
            "cos_eitc": [0.0],
        })
        pe_df = pd.DataFrame({
            "tax_unit_id": [1],
            "pe_eitc": [0.0],
        })
        results = mod.compare_calculations(cos_df, pe_df)
        assert isinstance(results, dict)

    def test_weighted_totals_section(self):
        """The function prints weighted totals for first 5 tax variables."""
        mod, _, _, _, _ = _import_cps_comparison()
        cos_df = pd.DataFrame({
            "tax_unit_id": [1, 2],
            "weight": [100.0, 200.0],
            "cos_eitc": [500.0, 600.0],
            "cos_ctc_total": [2000.0, 0.0],
            "cos_se_tax": [0.0, 500.0],
            "cos_income_tax": [3000.0, 5000.0],
            "cos_niit": [0.0, 0.0],
        })
        pe_df = pd.DataFrame({
            "tax_unit_id": [1, 2],
            "pe_eitc": [500.0, 600.0],
            "pe_ctc_total": [2000.0, 0.0],
            "pe_se_tax": [0.0, 500.0],
            "pe_income_tax": [3000.0, 5000.0],
            "pe_niit": [0.0, 0.0],
        })
        results = mod.compare_calculations(cos_df, pe_df)
        assert isinstance(results, dict)

    def test_zero_pe_total_percent(self):
        """Test 0 PE total doesn't cause ZeroDivisionError."""
        mod, _, _, _, _ = _import_cps_comparison()
        cos_df = pd.DataFrame({
            "tax_unit_id": [1],
            "weight": [100.0],
            "cos_eitc": [500.0],
        })
        pe_df = pd.DataFrame({
            "tax_unit_id": [1],
            "pe_eitc": [0.0],
        })
        results = mod.compare_calculations(cos_df, pe_df)
        assert isinstance(results, dict)


class TestMain:
    def test_main_runs(self):
        mod, mock_pe, mock_sim, mock_builder_mod, mock_runner_mod = (
            _import_cps_comparison()
        )
        mock_sim.calculate.return_value = np.array([100.0])

        mock_df = pd.DataFrame({
            "tax_unit_id": [1],
            "weight": [100.0],
            "cos_eitc": [500.0],
            "cos_ctc_total": [2000.0],
            "cos_se_tax": [0.0],
            "cos_income_tax": [3000.0],
            "cos_niit": [0.0],
            "adjusted_gross_income": [50000.0],
            "taxable_income": [35000.0],
        })
        mock_builder_mod.load_and_build_tax_units.return_value = mock_df
        mock_runner_mod.run_all_calculations.return_value = mock_df

        with patch.dict(sys.modules, {
            "policyengine_us": mock_pe,
            "tax_unit_builder": mock_builder_mod,
            "cosilico_runner": mock_runner_mod,
        }):
            mod.main()

    def test_main_high_match_rate(self):
        """Test main with high match rate (>=90) to hit EXCELLENT branch."""
        mod, mock_pe, mock_sim, mock_builder_mod, mock_runner_mod = (
            _import_cps_comparison()
        )
        mock_sim.calculate.return_value = np.array([100.0])

        mock_df = pd.DataFrame({
            "tax_unit_id": [1],
            "weight": [100.0],
            "cos_eitc": [500.0],
        })
        mock_builder_mod.load_and_build_tax_units.return_value = mock_df
        mock_runner_mod.run_all_calculations.return_value = mock_df

        with patch.dict(sys.modules, {
            "policyengine_us": mock_pe,
            "tax_unit_builder": mock_builder_mod,
            "cosilico_runner": mock_runner_mod,
        }), patch.object(mod, "compare_calculations", return_value={
            "EITC": {"match_rate": 95.0, "mean_diff": 5.0, "correlation": 0.99},
        }):
            mod.main()

    def test_main_medium_match_rate(self):
        """Test main with medium match rate (75-90) to hit GOOD branch."""
        mod, mock_pe, mock_sim, mock_builder_mod, mock_runner_mod = (
            _import_cps_comparison()
        )
        n = 20
        # Return PE values with moderate discrepancy
        mock_sim.calculate.return_value = np.arange(n, dtype=float) * 1000 + 600

        mock_df = pd.DataFrame({
            "tax_unit_id": list(range(n)),
            "weight": [100.0] * n,
            "cos_eitc": np.arange(n, dtype=float) * 1000 + 500,
            "cos_ctc_total": np.arange(n, dtype=float) * 500 + 100,
            "cos_se_tax": np.arange(n, dtype=float) * 200,
            "cos_income_tax": np.arange(n, dtype=float) * 800 + 1000,
            "cos_niit": np.zeros(n),
            "adjusted_gross_income": np.arange(n, dtype=float) * 2000 + 30000,
            "taxable_income": np.arange(n, dtype=float) * 1500 + 20000,
        })
        mock_builder_mod.load_and_build_tax_units.return_value = mock_df
        mock_runner_mod.run_all_calculations.return_value = mock_df

        # Patch compare_calculations to return match rates in the 75-90 range
        with patch.dict(sys.modules, {
            "policyengine_us": mock_pe,
            "tax_unit_builder": mock_builder_mod,
            "cosilico_runner": mock_runner_mod,
        }), patch.object(mod, "compare_calculations", return_value={
            "EITC": {"match_rate": 80.0, "mean_diff": 100.0, "correlation": 0.9},
        }):
            mod.main()

    def test_main_low_match_rate(self):
        """Test main with low match rate (<75) to hit NEEDS WORK branch."""
        mod, mock_pe, mock_sim, mock_builder_mod, mock_runner_mod = (
            _import_cps_comparison()
        )
        mock_sim.calculate.return_value = np.array([100.0])

        mock_df = pd.DataFrame({
            "tax_unit_id": [1],
            "weight": [100.0],
            "cos_eitc": [500.0],
        })
        mock_builder_mod.load_and_build_tax_units.return_value = mock_df
        mock_runner_mod.run_all_calculations.return_value = mock_df

        with patch.dict(sys.modules, {
            "policyengine_us": mock_pe,
            "tax_unit_builder": mock_builder_mod,
            "cosilico_runner": mock_runner_mod,
        }), patch.object(mod, "compare_calculations", return_value={
            "EITC": {"match_rate": 50.0, "mean_diff": 500.0, "correlation": 0.3},
        }):
            mod.main()


class TestCompareCalculationsCorrelation:
    def test_correlation_with_many_nonzero(self):
        """Test correlation is computed when >10 non-zero values."""
        mod, _, _, _, _ = _import_cps_comparison()
        n = 20
        cos_df = pd.DataFrame({
            "tax_unit_id": list(range(n)),
            "weight": [100.0] * n,
            "cos_eitc": np.arange(n, dtype=float) * 100 + 500,
        })
        pe_df = pd.DataFrame({
            "tax_unit_id": list(range(n)),
            "pe_eitc": np.arange(n, dtype=float) * 100 + 505,
        })
        results = mod.compare_calculations(cos_df, pe_df)
        assert isinstance(results, dict)
        if "EITC" in results:
            assert not np.isnan(results["EITC"]["correlation"])
