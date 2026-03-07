"""Tests for comparison/core.py module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cosilico_validators.comparison.core import (
    align_records,
    compare_records,
    generate_dashboard_json,
    load_cosilico_values,
    load_pe_values,
    run_full_comparison,
    run_variable_comparison,
)


class TestCompareRecords:
    def test_perfect_match(self):
        cos = np.array([100.0, 200.0, 300.0])
        pe = np.array([100.0, 200.0, 300.0])
        result = compare_records(cos, pe, tolerance=1.0)
        assert result["match_rate"] == 1.0
        assert result["n_records"] == 3
        assert result["n_matches"] == 3
        assert result["n_mismatches"] == 0

    def test_within_tolerance(self):
        cos = np.array([100.5, 200.5, 300.5])
        pe = np.array([100.0, 200.0, 300.0])
        result = compare_records(cos, pe, tolerance=1.0)
        assert result["match_rate"] == 1.0

    def test_mismatches(self):
        cos = np.array([100.0, 220.0, 300.0, 450.0])
        pe = np.array([100.0, 200.0, 300.0, 400.0])
        result = compare_records(cos, pe, tolerance=10.0)
        assert result["match_rate"] == 0.5
        assert result["n_mismatches"] == 2

    def test_error_percentiles(self):
        cos = np.array([100.0, 110.0, 120.0, 130.0, 200.0])
        pe = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        result = compare_records(cos, pe, tolerance=5.0)
        assert "error_percentiles" in result
        assert "p50" in result["error_percentiles"]
        assert "p95" in result["error_percentiles"]
        assert "p99" in result["error_percentiles"]

    def test_mean_absolute_error(self):
        cos = np.array([100.0, 200.0, 300.0, 400.0])
        pe = np.array([100.0, 220.0, 300.0, 450.0])
        result = compare_records(cos, pe, tolerance=10.0)
        assert result["mean_absolute_error"] == pytest.approx(17.5)

    def test_worst_mismatches(self):
        cos = np.array([100.0, 200.0, 1000.0, 400.0])
        pe = np.array([100.0, 200.0, 300.0, 400.0])
        result = compare_records(cos, pe, tolerance=10.0, top_n_mismatches=3)
        assert "worst_mismatches" in result
        assert len(result["worst_mismatches"]) >= 1
        worst = result["worst_mismatches"][0]
        assert worst["index"] == 2
        assert worst["cosilico"] == 1000.0
        assert worst["policyengine"] == 300.0
        assert worst["difference"] == 700.0


class TestLoadPeValues:
    def test_load_values(self):
        with (
            patch("cosilico_validators.comparison.core.HAS_POLICYENGINE", True),
            patch("cosilico_validators.comparison.core.Microsimulation") as mock_sim,
        ):
            mock_instance = MagicMock()
            mock_sim.return_value = mock_instance
            mock_instance.calculate.return_value = np.array([100.0, 200.0, 300.0])

            result = load_pe_values("income_tax", year=2024)
            assert len(result) == 3
            mock_instance.calculate.assert_called_with("income_tax", 2024)

    def test_load_values_with_ids(self):
        with (
            patch("cosilico_validators.comparison.core.HAS_POLICYENGINE", True),
            patch("cosilico_validators.comparison.core.Microsimulation") as mock_sim,
        ):
            mock_instance = MagicMock()
            mock_sim.return_value = mock_instance
            mock_instance.calculate.side_effect = [
                np.array([100.0, 200.0]),
                np.array([1, 2]),
            ]
            values, ids = load_pe_values("income_tax", year=2024, return_ids=True)
            assert len(values) == 2
            assert len(ids) == 2

    def test_requires_policyengine(self):
        from cosilico_validators.comparison.core import HAS_POLICYENGINE

        if not HAS_POLICYENGINE:
            with pytest.raises(ImportError):
                load_pe_values("eitc")


class TestLoadCosilicoValues:
    def test_load_values(self, tmp_path):
        # Create the directory so exists() returns True
        data_dir = tmp_path / "CosilicoAI" / "cosilico-data-sources" / "micro" / "us"
        data_dir.mkdir(parents=True)

        import pandas as pd

        mock_df = pd.DataFrame(
            {
                "cos_income_tax": [100.0, 200.0, 300.0],
                "tax_unit_id": [1, 2, 3],
            }
        )

        with (
            patch("pathlib.Path.home", return_value=tmp_path),
            patch.dict(
                "sys.modules",
                {
                    "tax_unit_builder": MagicMock(),
                    "cosilico_runner": MagicMock(),
                },
            ),
        ):
            import sys

            mock_builder = sys.modules["tax_unit_builder"]
            mock_runner = sys.modules["cosilico_runner"]
            mock_builder.load_and_build_tax_units.return_value = mock_df
            mock_runner.run_all_calculations.return_value = mock_df

            result = load_cosilico_values("income_tax", year=2024)
            assert len(result) == 3

    def test_missing_data_sources(self, tmp_path):
        with (
            patch("pathlib.Path.home", return_value=tmp_path),
            pytest.raises(ImportError, match="cosilico-data-sources not found"),
        ):
            load_cosilico_values("eitc")

    def test_load_values_with_ids(self, tmp_path):
        """Test return_ids=True path."""
        data_dir = tmp_path / "CosilicoAI" / "cosilico-data-sources" / "micro" / "us"
        data_dir.mkdir(parents=True)

        import pandas as pd

        mock_df = pd.DataFrame(
            {
                "cos_income_tax": [100.0, 200.0, 300.0],
                "tax_unit_id": [1, 2, 3],
            }
        )

        with (
            patch("pathlib.Path.home", return_value=tmp_path),
            patch.dict(
                "sys.modules",
                {
                    "tax_unit_builder": MagicMock(),
                    "cosilico_runner": MagicMock(),
                },
            ),
        ):
            import sys

            mock_builder = sys.modules["tax_unit_builder"]
            mock_runner = sys.modules["cosilico_runner"]
            mock_builder.load_and_build_tax_units.return_value = mock_df
            mock_runner.run_all_calculations.return_value = mock_df

            values, ids = load_cosilico_values("income_tax", year=2024, return_ids=True)
            assert len(values) == 3
            assert len(ids) == 3
            np.testing.assert_array_equal(ids, [1, 2, 3])

    def test_load_variable_not_found(self, tmp_path):
        """Test ValueError when variable column is not in dataframe."""
        data_dir = tmp_path / "CosilicoAI" / "cosilico-data-sources" / "micro" / "us"
        data_dir.mkdir(parents=True)

        import pandas as pd

        mock_df = pd.DataFrame(
            {
                "cos_eitc": [100.0],
                "tax_unit_id": [1],
            }
        )

        with (
            patch("pathlib.Path.home", return_value=tmp_path),
            patch.dict(
                "sys.modules",
                {
                    "tax_unit_builder": MagicMock(),
                    "cosilico_runner": MagicMock(),
                },
            ),
        ):
            import sys

            mock_builder = sys.modules["tax_unit_builder"]
            mock_runner = sys.modules["cosilico_runner"]
            mock_builder.load_and_build_tax_units.return_value = mock_df
            mock_runner.run_all_calculations.return_value = mock_df

            with pytest.raises(ValueError, match="not found"):
                load_cosilico_values("nonexistent_xyz", year=2024)


class TestRunVariableComparison:
    def test_run_comparison(self):
        with (
            patch("cosilico_validators.comparison.core.load_pe_values") as mock_pe,
            patch("cosilico_validators.comparison.core.load_cosilico_values") as mock_cos,
        ):
            mock_pe.return_value = (np.array([100.0, 200.0, 300.0]), np.array([1, 2, 3]))
            mock_cos.return_value = (np.array([100.0, 200.0, 300.0]), np.array([1, 2, 3]))

            result = run_variable_comparison("income_tax", year=2024)
            assert result["variable"] == "income_tax"
            assert result["match_rate"] == 1.0
            assert result["matched_records"] == 3


class TestGenerateDashboardJson:
    def test_dashboard_structure(self):
        results = [
            {"variable": "income_tax", "match_rate": 0.95, "n_records": 30000, "mean_absolute_error": 50.0},
            {"variable": "eitc", "match_rate": 0.99, "n_records": 30000, "mean_absolute_error": 5.0},
        ]
        dashboard = generate_dashboard_json(results, year=2024)
        assert "metadata" in dashboard
        assert "variables" in dashboard
        assert len(dashboard["variables"]) == 2
        assert dashboard["variables"][0]["variable"] == "income_tax"

    def test_dashboard_includes_summary(self):
        results = [
            {"variable": "v1", "match_rate": 0.90, "n_records": 100, "mean_absolute_error": 10.0},
            {"variable": "v2", "match_rate": 0.80, "n_records": 100, "mean_absolute_error": 20.0},
        ]
        dashboard = generate_dashboard_json(results, year=2024)
        assert "summary" in dashboard
        assert dashboard["summary"]["overall_match_rate"] == pytest.approx(0.85)
        assert dashboard["summary"]["total_records"] == 200


class TestRunFullComparison:
    def test_run_full_default(self):
        with patch("cosilico_validators.comparison.core.run_variable_comparison") as mock_rv:
            mock_rv.return_value = {
                "variable": "eitc",
                "match_rate": 0.95,
                "n_records": 1000,
                "mean_absolute_error": 50,
            }
            result = run_full_comparison(year=2024)
            assert "variables" in result
            assert "summary" in result

    def test_run_full_with_specific_variables(self):
        with patch("cosilico_validators.comparison.core.run_variable_comparison") as mock_rv:
            mock_rv.return_value = {
                "variable": "eitc",
                "match_rate": 0.95,
                "n_records": 1000,
                "mean_absolute_error": 50,
            }
            result = run_full_comparison(variables=["eitc"], year=2024)
            assert len(result["variables"]) == 1

    def test_run_full_handles_error(self):
        with patch("cosilico_validators.comparison.core.run_variable_comparison", side_effect=Exception("fail")):
            result = run_full_comparison(year=2024)
            # Should handle errors gracefully
            assert "variables" in result
            for v in result["variables"]:
                assert "error" in v


class TestAlignRecords:
    def test_align_matching_ids(self):
        cos_values = np.array([100.0, 200.0, 300.0])
        cos_ids = np.array([1, 2, 3])
        pe_values = np.array([100.0, 200.0, 300.0])
        pe_ids = np.array([1, 2, 3])
        aligned_cos, aligned_pe, matched_ids = align_records(cos_values, cos_ids, pe_values, pe_ids)
        assert len(aligned_cos) == 3
        assert len(aligned_pe) == 3
        np.testing.assert_array_equal(aligned_cos, cos_values)

    def test_align_partial_overlap(self):
        cos_values = np.array([100.0, 200.0, 300.0])
        cos_ids = np.array([1, 2, 3])
        pe_values = np.array([200.0, 300.0, 400.0])
        pe_ids = np.array([2, 3, 4])
        aligned_cos, aligned_pe, matched_ids = align_records(cos_values, cos_ids, pe_values, pe_ids)
        assert len(aligned_cos) == 2
        assert len(aligned_pe) == 2
        np.testing.assert_array_equal(matched_ids, [2, 3])

    def test_align_no_overlap(self):
        cos_values = np.array([100.0])
        cos_ids = np.array([1])
        pe_values = np.array([200.0])
        pe_ids = np.array([2])
        with pytest.raises(ValueError, match="No matching"):
            align_records(cos_values, cos_ids, pe_values, pe_ids)
