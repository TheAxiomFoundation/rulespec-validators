"""Tests for record-by-record Cosilico vs PolicyEngine comparison.

TDD: Write tests first for what the harness should do.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestRecordComparison:
    """Test record-by-record comparison of Cosilico vs PolicyEngine."""

    def test_comparison_returns_match_rate(self):
        """Comparison should return match rate between 0 and 1."""
        from cosilico_validators.comparison import compare_records

        # Mock: 100 records, Cosilico matches PE exactly
        cosilico_values = np.array([100.0, 200.0, 300.0, 0.0, 500.0])
        pe_values = np.array([100.0, 200.0, 300.0, 0.0, 500.0])

        result = compare_records(cosilico_values, pe_values, tolerance=1.0)

        assert result["match_rate"] == 1.0
        assert result["n_records"] == 5
        assert result["n_matches"] == 5

    def test_comparison_with_small_differences(self):
        """Small differences within tolerance should still match."""
        from cosilico_validators.comparison import compare_records

        cosilico_values = np.array([100.0, 200.5, 300.0])
        pe_values = np.array([100.0, 200.0, 300.5])  # 0.5 diff

        result = compare_records(cosilico_values, pe_values, tolerance=1.0)

        assert result["match_rate"] == 1.0  # All within $1 tolerance

    def test_comparison_with_mismatches(self):
        """Differences beyond tolerance should be mismatches."""
        from cosilico_validators.comparison import compare_records

        cosilico_values = np.array([100.0, 200.0, 300.0, 400.0])
        pe_values = np.array([100.0, 220.0, 300.0, 450.0])  # 2 mismatches

        result = compare_records(cosilico_values, pe_values, tolerance=10.0)

        assert result["match_rate"] == 0.5  # 2/4 match
        assert result["n_mismatches"] == 2
        assert result["mean_absolute_error"] == pytest.approx(17.5)  # (0+20+0+50)/4

    def test_comparison_returns_error_distribution(self):
        """Should return percentiles of absolute errors."""
        from cosilico_validators.comparison import compare_records

        cosilico_values = np.array([100.0, 110.0, 120.0, 130.0, 200.0])
        pe_values = np.array([100.0, 100.0, 100.0, 100.0, 100.0])

        result = compare_records(cosilico_values, pe_values, tolerance=5.0)

        assert "error_percentiles" in result
        assert "p50" in result["error_percentiles"]
        assert "p95" in result["error_percentiles"]
        assert "p99" in result["error_percentiles"]

    def test_comparison_identifies_worst_mismatches(self):
        """Should return indices and values of worst mismatches."""
        from cosilico_validators.comparison import compare_records

        cosilico_values = np.array([100.0, 200.0, 1000.0, 400.0])  # idx 2 is way off
        pe_values = np.array([100.0, 200.0, 300.0, 400.0])

        result = compare_records(cosilico_values, pe_values, tolerance=10.0, top_n_mismatches=3)

        assert "worst_mismatches" in result
        assert len(result["worst_mismatches"]) >= 1
        # Worst mismatch should be index 2 with diff of 700
        worst = result["worst_mismatches"][0]
        assert worst["index"] == 2
        assert worst["cosilico"] == 1000.0
        assert worst["policyengine"] == 300.0
        assert worst["difference"] == 700.0


class TestCPSComparison:
    """Test running comparison on actual CPS data."""

    def test_load_pe_values_for_variable(self):
        """Should load PolicyEngine values for a variable across CPS."""
        from cosilico_validators.comparison import load_pe_values

        with patch("cosilico_validators.comparison.core.HAS_POLICYENGINE", True), \
             patch("cosilico_validators.comparison.core.Microsimulation") as mock_sim:
            mock_instance = MagicMock()
            mock_sim.return_value = mock_instance
            mock_instance.calculate.return_value = np.array([100.0, 200.0, 300.0])

            values = load_pe_values("income_tax", year=2024)

            assert len(values) == 3
            mock_instance.calculate.assert_called_with("income_tax", 2024)

    def test_load_cosilico_values_for_variable(self, tmp_path):
        """Should load Cosilico-computed values for a variable across CPS."""
        import pandas as pd

        from cosilico_validators.comparison import load_cosilico_values

        # Create data sources directory
        data_dir = tmp_path / "CosilicoAI" / "cosilico-data-sources" / "micro" / "us"
        data_dir.mkdir(parents=True)

        mock_df = pd.DataFrame({
            "cos_income_tax": [100.0, 200.0, 300.0],
            "tax_unit_id": [1, 2, 3],
        })

        with patch("pathlib.Path.home", return_value=tmp_path), \
             patch.dict("sys.modules", {
                "tax_unit_builder": MagicMock(),
                "cosilico_runner": MagicMock(),
             }):
            import sys
            mock_builder = sys.modules["tax_unit_builder"]
            mock_runner = sys.modules["cosilico_runner"]
            mock_builder.load_and_build_tax_units.return_value = mock_df
            mock_runner.run_all_calculations.return_value = mock_df

            values = load_cosilico_values("income_tax", year=2024)

            assert len(values) == 3
            mock_builder.load_and_build_tax_units.assert_called_with(2024)
            mock_runner.run_all_calculations.assert_called_with(mock_df, 2024)

    def test_run_comparison_for_variable(self):
        """Should run full comparison for a single variable."""
        from cosilico_validators.comparison import run_variable_comparison

        with patch("cosilico_validators.comparison.core.load_pe_values") as mock_pe, \
             patch("cosilico_validators.comparison.core.load_cosilico_values") as mock_cos:
            # Return (values, ids) tuples since return_ids=True
            mock_pe.return_value = (np.array([100.0, 200.0, 300.0]), np.array([1, 2, 3]))
            mock_cos.return_value = (np.array([100.0, 200.0, 300.0]), np.array([1, 2, 3]))

            result = run_variable_comparison("income_tax", year=2024)

            assert result["variable"] == "income_tax"
            assert result["match_rate"] == 1.0
            assert result["matched_records"] == 3


class TestComparisonDashboard:
    """Test dashboard output format."""

    def test_dashboard_json_structure(self):
        """Dashboard JSON should have required structure."""
        from cosilico_validators.comparison import generate_dashboard_json

        results = [
            {
                "variable": "income_tax",
                "match_rate": 0.95,
                "n_records": 30000,
                "mean_absolute_error": 50.0,
            },
            {
                "variable": "eitc",
                "match_rate": 0.99,
                "n_records": 30000,
                "mean_absolute_error": 5.0,
            },
        ]

        dashboard = generate_dashboard_json(results, year=2024)

        assert "metadata" in dashboard
        assert "variables" in dashboard
        assert len(dashboard["variables"]) == 2
        assert dashboard["variables"][0]["variable"] == "income_tax"

    def test_dashboard_includes_overall_summary(self):
        """Dashboard should include overall summary stats."""
        from cosilico_validators.comparison import generate_dashboard_json

        results = [
            {"variable": "v1", "match_rate": 0.90, "n_records": 100, "mean_absolute_error": 10.0},
            {"variable": "v2", "match_rate": 0.80, "n_records": 100, "mean_absolute_error": 20.0},
        ]

        dashboard = generate_dashboard_json(results, year=2024)

        assert "summary" in dashboard
        assert dashboard["summary"]["overall_match_rate"] == pytest.approx(0.85)  # avg
        assert dashboard["summary"]["total_records"] == 200
