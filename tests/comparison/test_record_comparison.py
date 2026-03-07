"""Tests for comparison/record_comparison.py module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from cosilico_validators.comparison.record_comparison import (
    RecordComparison,
    _create_pe_situation,
    _safe_float,
    _safe_int,
    print_comparison,
)


class TestRecordComparison:
    def _make_comparison(self):
        n = 100
        return RecordComparison(
            variable="eitc",
            n_records=n,
            cosilico=np.ones(n) * 500,
            policyengine=np.ones(n) * 510,
            taxsim=np.ones(n) * 505,
            taxcalc=np.zeros(n),
            weights=np.ones(n) * 100,
            cosilico_ms=50.0,
            policyengine_ms=200.0,
            taxsim_ms=150.0,
            taxcalc_ms=0.0,
        )

    def test_weighted_totals(self):
        comp = self._make_comparison()
        totals = comp.weighted_totals
        assert totals["cosilico"] == pytest.approx(500 * 100 * 100)
        assert totals["policyengine"] == pytest.approx(510 * 100 * 100)

    def test_mean_abs_diff_vs_pe(self):
        comp = self._make_comparison()
        diffs = comp.mean_abs_diff_vs_pe
        assert diffs["cosilico"] == pytest.approx(10.0)
        assert diffs["taxsim"] == pytest.approx(5.0)
        assert diffs["taxcalc"] == pytest.approx(510.0)

    def test_match_rate_vs_pe(self):
        comp = self._make_comparison()
        rates = comp.match_rate_vs_pe
        # With tolerance=1.0, 10-dollar diff shouldn't match
        assert isinstance(rates, dict)


class TestSafeConversions:
    def test_safe_int_normal(self):
        assert _safe_int(5) == 5
        assert _safe_int(5.0) == 5

    def test_safe_int_nan(self):
        assert _safe_int(float("nan")) == 0
        assert _safe_int(float("nan"), default=42) == 42

    def test_safe_int_none(self):
        assert _safe_int(None) == 0
        assert _safe_int(None, default=10) == 10

    def test_safe_float_normal(self):
        assert _safe_float(5.5) == 5.5

    def test_safe_float_nan(self):
        assert _safe_float(float("nan")) == 0.0
        assert _safe_float(float("nan"), default=1.5) == 1.5

    def test_safe_float_none(self):
        assert _safe_float(None) == 0.0


class TestCreatePeSituation:
    def test_single_filer(self):
        row = pd.Series(
            {
                "is_joint": False,
                "num_dependents": 0,
                "head_age": 35,
                "wage_income": 50000,
                "self_employment_income": 0,
                "social_security_income": 0,
                "interest_income": 0,
                "dividend_income": 0,
                "rental_income": 0,
                "unemployment_compensation": 0,
                "num_eitc_children": 0,
                "num_ctc_children": 0,
                "num_other_dependents": 0,
            }
        )
        situation = _create_pe_situation(row, 2024)
        assert "people" in situation
        assert "head" in situation["people"]
        assert "tax_units" in situation
        assert "households" in situation

    def test_joint_filer(self):
        row = pd.Series(
            {
                "is_joint": True,
                "num_dependents": 1,
                "head_age": 40,
                "spouse_age": 38,
                "wage_income": 80000,
                "self_employment_income": 0,
                "social_security_income": 0,
                "interest_income": 0,
                "dividend_income": 0,
                "rental_income": 0,
                "unemployment_compensation": 0,
                "num_eitc_children": 1,
                "num_ctc_children": 1,
                "num_other_dependents": 0,
            }
        )
        situation = _create_pe_situation(row, 2024)
        assert "spouse" in situation["people"]
        assert "dep1" in situation["people"]

    def test_multiple_dependents(self):
        row = pd.Series(
            {
                "is_joint": False,
                "num_dependents": 3,
                "head_age": 35,
                "wage_income": 30000,
                "self_employment_income": 0,
                "social_security_income": 0,
                "interest_income": 0,
                "dividend_income": 0,
                "rental_income": 0,
                "unemployment_compensation": 0,
                "num_eitc_children": 2,
                "num_ctc_children": 2,
                "num_other_dependents": 1,
            }
        )
        situation = _create_pe_situation(row, 2024)
        assert "dep1" in situation["people"]
        assert "dep2" in situation["people"]
        assert "dep3" in situation["people"]

    def test_ctc_only_children(self):
        """Test CTC-only children path where n_ctc_children > n_eitc_children."""
        row = pd.Series(
            {
                "is_joint": False,
                "num_dependents": 3,
                "head_age": 35,
                "wage_income": 30000,
                "self_employment_income": 0,
                "social_security_income": 0,
                "interest_income": 0,
                "dividend_income": 0,
                "rental_income": 0,
                "unemployment_compensation": 0,
                "num_eitc_children": 1,
                "num_ctc_children": 3,  # More CTC children than EITC children
                "num_other_dependents": 0,
            }
        )
        situation = _create_pe_situation(row, 2024)
        assert "dep1" in situation["people"]  # EITC child
        assert "dep2" in situation["people"]  # CTC-only child
        assert "dep3" in situation["people"]  # CTC-only child
        # All three should have age 10 (young children)
        for dep_name in ["dep1", "dep2", "dep3"]:
            assert situation["people"][dep_name]["age"]["2024"] == 10


class TestPrintComparison:
    def test_print(self):
        n = 10
        results = {
            "eitc": RecordComparison(
                variable="eitc",
                n_records=n,
                cosilico=np.ones(n) * 500,
                policyengine=np.ones(n) * 510,
                taxsim=np.ones(n) * 505,
                taxcalc=np.zeros(n),
                weights=np.ones(n),
                cosilico_ms=10,
                policyengine_ms=20,
                taxsim_ms=15,
                taxcalc_ms=0,
            )
        }
        # Should not raise
        print_comparison(results)


class TestLoadCpsInputs:
    def test_load_cps_inputs(self):
        from cosilico_validators.comparison.record_comparison import load_cps_inputs

        mock_builder = MagicMock()
        mock_df = pd.DataFrame({"weight": [100.0], "earned_income": [50000.0]})
        mock_builder.load_and_build_tax_units.return_value = mock_df

        with patch("pathlib.Path.home", return_value=MagicMock()):
            import sys

            with patch.dict(sys.modules, {"tax_unit_builder": mock_builder}):
                result = load_cps_inputs(year=2024)
                assert isinstance(result, pd.DataFrame)


class TestRunCosilico:
    def test_run_cosilico(self):
        from cosilico_validators.comparison.record_comparison import run_cosilico

        mock_runner = MagicMock()
        input_df = pd.DataFrame({"weight": [100.0], "earned_income": [50000.0]})
        mock_runner.run_all_calculations.return_value = input_df.copy()

        with patch("pathlib.Path.home", return_value=MagicMock()):
            import sys

            with patch.dict(sys.modules, {"cosilico_runner": mock_runner}):
                result_df, elapsed_ms = run_cosilico(input_df, year=2024)
                assert isinstance(result_df, pd.DataFrame)
                assert elapsed_ms >= 0


class TestRunPolicyengine:
    def test_run_policyengine(self):
        import sys

        from cosilico_validators.comparison.record_comparison import run_policyengine

        mock_pe = MagicMock()
        mock_sim = MagicMock()
        mock_pe.Simulation.return_value = mock_sim
        mock_sim.calculate.return_value = np.array([500.0])

        input_df = pd.DataFrame(
            {
                "is_joint": [False],
                "num_dependents": [0],
                "head_age": [35],
                "wage_income": [50000],
                "self_employment_income": [0],
                "social_security_income": [0],
                "interest_income": [0],
                "dividend_income": [0],
                "rental_income": [0],
                "unemployment_compensation": [0],
                "num_eitc_children": [0],
                "num_ctc_children": [0],
                "num_other_dependents": [0],
            }
        )

        with patch.dict(sys.modules, {"policyengine_us": mock_pe}):
            result_df, elapsed_ms = run_policyengine(input_df, year=2024)
            assert isinstance(result_df, pd.DataFrame)
            assert elapsed_ms >= 0
            assert "eitc" in result_df.columns


class TestRunTaxsim:
    def test_run_taxsim_success(self):
        from cosilico_validators.comparison.record_comparison import run_taxsim

        input_df = pd.DataFrame(
            {
                "weight": [100.0],
                "is_joint": [False],
                "head_age": [35],
                "num_dependents": [0],
                "wage_income": [50000],
                "dividend_income": [0],
                "interest_income": [0],
                "rental_income": [0],
                "social_security_income": [0],
                "self_employment_income": [0],
            }
        )

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "taxsimid,year,v25,v22,actc,v19,v10\n1,2024,0.00,0.00,0.00,3000.00,50000.00"

        with (
            patch(
                "cosilico_validators.comparison.multi_validator.get_taxsim_executable_path",
                return_value="/tmp/taxsim35",
            ),
            patch("subprocess.run", return_value=mock_result),
        ):
            result_df, elapsed_ms = run_taxsim(input_df, year=2024)
            assert isinstance(result_df, pd.DataFrame)
            assert elapsed_ms >= 0
            assert "eitc" in result_df.columns

    def test_run_taxsim_failure(self):
        from cosilico_validators.comparison.record_comparison import run_taxsim

        input_df = pd.DataFrame(
            {
                "weight": [100.0],
                "is_joint": [False],
                "head_age": [35],
                "num_dependents": [0],
                "wage_income": [50000],
            }
        )

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "TAXSIM error"

        with (
            patch(
                "cosilico_validators.comparison.multi_validator.get_taxsim_executable_path",
                return_value="/tmp/taxsim35",
            ),
            patch("subprocess.run", return_value=mock_result),
            pytest.raises(RuntimeError, match="TAXSIM failed"),
        ):
            run_taxsim(input_df, year=2024)

    def test_run_taxsim_joint_filer(self):
        from cosilico_validators.comparison.record_comparison import run_taxsim

        input_df = pd.DataFrame(
            {
                "weight": [100.0],
                "is_joint": [True],
                "head_age": [40],
                "spouse_age": [38],
                "num_dependents": [1],
                "wage_income": [80000],
                "dividend_income": [5000],
                "interest_income": [3000],
                "rental_income": [0],
                "social_security_income": [0],
                "self_employment_income": [10000],
            }
        )

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "taxsimid,year,v25,v22,actc,v19,v10\n1,2024,0.00,2000.00,0.00,5000.00,98000.00"

        with (
            patch(
                "cosilico_validators.comparison.multi_validator.get_taxsim_executable_path",
                return_value="/tmp/taxsim35",
            ),
            patch("subprocess.run", return_value=mock_result),
        ):
            result_df, elapsed_ms = run_taxsim(input_df, year=2024)
            assert len(result_df) == 1


class TestCompareRecords:
    def test_compare_records_full(self):
        from cosilico_validators.comparison.record_comparison import (
            compare_records as cr_compare_records,
        )

        mock_df = pd.DataFrame(
            {
                "weight": [100.0],
                "is_joint": [False],
                "head_age": [35],
                "num_dependents": [0],
                "wage_income": [50000],
                "self_employment_income": [0],
                "social_security_income": [0],
                "interest_income": [0],
                "dividend_income": [0],
                "rental_income": [0],
                "unemployment_compensation": [0],
                "num_eitc_children": [0],
                "num_ctc_children": [0],
                "num_other_dependents": [0],
            }
        )

        cos_result = pd.DataFrame({"eitc": [500.0]})
        pe_result = pd.DataFrame({"eitc": [510.0]})
        ts_result = pd.DataFrame({"eitc": [505.0]})

        with (
            patch(
                "cosilico_validators.comparison.record_comparison.load_cps_inputs",
                return_value=mock_df,
            ),
            patch(
                "cosilico_validators.comparison.record_comparison.run_cosilico",
                return_value=(cos_result, 50.0),
            ),
            patch(
                "cosilico_validators.comparison.record_comparison.run_policyengine",
                return_value=(pe_result, 200.0),
            ),
            patch(
                "cosilico_validators.comparison.record_comparison.run_taxsim",
                return_value=(ts_result, 150.0),
            ),
        ):
            results = cr_compare_records(year=2024, variables=["eitc"], sample_size=1)
            assert "eitc" in results
            assert results["eitc"].n_records == 1

    def test_compare_records_default_variables(self):
        from cosilico_validators.comparison.record_comparison import (
            compare_records as cr_compare_records,
        )

        mock_df = pd.DataFrame(
            {
                "weight": [100.0, 200.0],
            }
        )

        cos_result = pd.DataFrame(
            {
                "eitc": [500.0, 600.0],
                "non_refundable_ctc": [0.0, 2000.0],
                "refundable_ctc": [0.0, 0.0],
            }
        )
        pe_result = cos_result.copy()
        ts_result = cos_result.copy()

        with (
            patch(
                "cosilico_validators.comparison.record_comparison.load_cps_inputs",
                return_value=mock_df,
            ),
            patch(
                "cosilico_validators.comparison.record_comparison.run_cosilico",
                return_value=(cos_result, 50.0),
            ),
            patch(
                "cosilico_validators.comparison.record_comparison.run_policyengine",
                return_value=(pe_result, 200.0),
            ),
            patch(
                "cosilico_validators.comparison.record_comparison.run_taxsim",
                return_value=(ts_result, 150.0),
            ),
        ):
            results = cr_compare_records(year=2024)
            assert "eitc" in results
            assert "non_refundable_ctc" in results
            assert "refundable_ctc" in results
