"""Tests for comparison/aligned.py module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cosilico_validators.comparison.aligned import (
    CommonDataset,
    ComparisonResult,
    _var_exists,
    compare_variable,
)


def _make_common_dataset(n=100):
    """Create a CommonDataset with all required fields."""
    return CommonDataset(
        tax_unit_id=np.arange(n),
        weight=np.ones(n) * 100,
        is_joint=np.zeros(n, dtype=bool),
        filing_status=np.array(["SINGLE"] * n),
        earned_income=np.ones(n) * 50000,
        wages=np.ones(n) * 50000,
        self_employment_income=np.zeros(n),
        partnership_s_corp_income=np.zeros(n),
        farm_income=np.zeros(n),
        interest_income=np.zeros(n),
        dividend_income=np.zeros(n),
        capital_gains=np.zeros(n),
        rental_income=np.zeros(n),
        taxable_social_security=np.zeros(n),
        pension_income=np.zeros(n),
        taxable_unemployment=np.zeros(n),
        retirement_distributions=np.zeros(n),
        miscellaneous_income=np.zeros(n),
        other_income=np.zeros(n),
        investment_income=np.zeros(n),
        adjusted_gross_income=np.ones(n) * 50000,
        taxable_income=np.ones(n) * 35000,
        eitc_child_count=np.zeros(n),
        ctc_child_count=np.zeros(n),
        head_age=np.ones(n) * 30,
        spouse_age=np.zeros(n),
        head_is_blind=np.zeros(n, dtype=bool),
        spouse_is_blind=np.zeros(n, dtype=bool),
        head_is_dependent=np.zeros(n, dtype=bool),
        cdcc_qualifying_individuals=np.zeros(n),
        childcare_expenses=np.zeros(n),
        self_employment_tax_deduction=np.zeros(n),
        self_employed_health_insurance_deduction=np.zeros(n),
        educator_expense_deduction=np.zeros(n),
        loss_deduction=np.zeros(n),
        self_employed_pension_deduction=np.zeros(n),
        ira_deduction=np.zeros(n),
        hsa_deduction=np.zeros(n),
        student_loan_interest_deduction=np.zeros(n),
        above_the_line_deductions_total=np.zeros(n),
    )


class TestComparisonResult:
    def test_creation(self):
        n = 100
        result = ComparisonResult(
            variable="eitc",
            match_rate=0.95,
            mean_absolute_error=50.0,
            n_records=n,
            cosilico_total=60e9,
            policyengine_total=62e9,
            cosilico_values=np.ones(n) * 600,
            policyengine_values=np.ones(n) * 620,
            error_percentiles={"p50": 10, "p95": 100, "p99": 500, "max": 1000},
        )
        assert result.variable == "eitc"
        assert result.match_rate == 0.95
        assert result.n_records == n
        assert result.cosilico_total == 60e9


class TestCommonDataset:
    def test_creation(self):
        n = 100
        ds = _make_common_dataset(n)
        assert ds.n_records == n

    def test_n_records_property(self):
        ds = _make_common_dataset(50)
        assert ds.n_records == 50


class TestCompareVariable:
    def test_perfect_match(self):
        ds = _make_common_dataset(1000)
        pe_values = np.ones(ds.n_records) * 1000

        def cos_func(dataset):
            return np.ones(dataset.n_records) * 1000

        result = compare_variable(ds, cos_func, pe_values, "eitc")
        assert result.match_rate == 1.0
        assert result.mean_absolute_error == 0.0

    def test_partial_match(self):
        ds = _make_common_dataset(1000)
        pe_values = np.ones(ds.n_records) * 1000
        cos_values = np.ones(ds.n_records) * 1000
        cos_values[:500] = 2000

        def cos_func(dataset, v=cos_values):
            return v

        result = compare_variable(ds, cos_func, pe_values, "eitc")
        assert result.match_rate < 1.0
        assert result.mean_absolute_error > 0

    def test_result_has_percentiles(self):
        ds = _make_common_dataset(1000)
        pe_values = np.ones(ds.n_records) * 1000
        np.random.seed(42)

        def cos_func(dataset):
            return np.random.uniform(900, 1100, dataset.n_records)

        result = compare_variable(ds, cos_func, pe_values, "eitc")
        assert "p50" in result.error_percentiles
        assert "p90" in result.error_percentiles
        assert "p95" in result.error_percentiles
        assert "p99" in result.error_percentiles
        assert "max" in result.error_percentiles

    def test_result_totals(self):
        ds = _make_common_dataset(1000)
        pe_values = np.ones(ds.n_records) * 1000
        cos_values = np.ones(ds.n_records) * 1000

        def cos_func(dataset, v=cos_values):
            return v

        result = compare_variable(ds, cos_func, pe_values, "eitc")
        expected_total = np.sum(ds.weight * 1000)
        assert result.cosilico_total == pytest.approx(expected_total)
        assert result.policyengine_total == pytest.approx(expected_total)

    def test_custom_tolerance(self):
        ds = _make_common_dataset(100)
        pe_values = np.ones(ds.n_records) * 1000
        cos_values = np.ones(ds.n_records) * 1005

        def cos_func(dataset, v=cos_values):
            return v

        # With tolerance=1, 5-dollar diff should not match
        result_strict = compare_variable(ds, cos_func, pe_values, "eitc", tolerance=1.0)
        assert result_strict.match_rate == 0.0
        # With tolerance=10, 5-dollar diff should match
        result_loose = compare_variable(ds, cos_func, pe_values, "eitc", tolerance=10.0)
        assert result_loose.match_rate == 1.0

    def test_result_has_values(self):
        ds = _make_common_dataset(100)
        pe_values = np.ones(ds.n_records) * 1000

        def cos_func(dataset):
            return np.ones(dataset.n_records) * 1000

        result = compare_variable(ds, cos_func, pe_values, "eitc")
        assert len(result.cosilico_values) == ds.n_records
        assert len(result.policyengine_values) == ds.n_records


class TestVarExists:
    def test_var_exists_true(self):
        mock_sim = MagicMock()
        mock_sim.calculate.return_value = np.zeros(10)
        assert _var_exists(mock_sim, "test_var", 2024) is True

    def test_var_exists_false(self):
        mock_sim = MagicMock()
        mock_sim.calculate.side_effect = Exception("not found")
        assert _var_exists(mock_sim, "test_var", 2024) is False


class TestLoadCommonDataset:
    def test_requires_policyengine(self):
        from cosilico_validators.comparison.aligned import HAS_POLICYENGINE, load_common_dataset

        if not HAS_POLICYENGINE:
            with pytest.raises(ImportError):
                load_common_dataset()

    def test_load_with_mocked_pe(self):
        """Test load_common_dataset with fully mocked PolicyEngine."""
        import sys

        mock_pe = MagicMock()
        mock_sim = MagicMock()
        mock_pe.Microsimulation.return_value = mock_sim

        # Create arrays of various sizes
        n_tax_units = 5
        n_persons = 8

        tu_ids = np.arange(n_tax_units)
        person_tu_ids = np.array([0, 0, 1, 1, 2, 3, 3, 4])

        def mock_calculate(var, year):
            # Tax unit-level variables
            tu_vars = {
                "tax_unit_id": tu_ids,
                "tax_unit_weight": np.ones(n_tax_units) * 100,
                "tax_unit_is_joint": np.array([False, True, False, True, False]),
                "filing_status": np.array(["SINGLE", "JOINT", "HEAD_OF_HOUSEHOLD", "JOINT", "SINGLE"]),
                "age_head": np.array([35, 40, 30, 45, 28]),
                "age_spouse": np.array([0, 38, 0, 43, 0]),
                "tax_unit_earned_income": np.ones(n_tax_units) * 50000,
                "net_investment_income": np.zeros(n_tax_units),
                "adjusted_gross_income": np.ones(n_tax_units) * 50000,
                "taxable_income": np.ones(n_tax_units) * 35000,
                "eitc_child_count": np.zeros(n_tax_units),
            }
            # Person-level variables
            # Person 5 (idx 5) is head of tax unit 3, is blind AND dependent
            # Person 6 (idx 6) is spouse of tax unit 3, is blind
            person_vars = {
                "is_blind": np.array([False, False, False, False, False, True, True, False]),
                "is_tax_unit_head": np.array([True, False, True, False, True, True, False, True]),
                "is_tax_unit_spouse": np.array([False, True, False, True, False, False, True, False]),
                "person_tax_unit_id": person_tu_ids,
                "is_tax_unit_dependent": np.array([False, False, False, False, False, True, False, False]),
                "irs_employment_income": np.ones(n_persons) * 50000,
            }
            # Check in both dicts
            if var in tu_vars:
                return tu_vars[var]
            if var in person_vars:
                return person_vars[var]
            # For any other variable, return person-level zeros
            if "tax_unit" in var or "above_the_line" in var:
                return np.zeros(n_tax_units)
            return np.zeros(n_persons)

        mock_sim.calculate.side_effect = mock_calculate

        with (
            patch.dict(sys.modules, {"policyengine_us": mock_pe}),
            patch("cosilico_validators.comparison.aligned.HAS_POLICYENGINE", True),
            patch("cosilico_validators.comparison.aligned.Microsimulation", mock_pe.Microsimulation),
        ):
            from cosilico_validators.comparison.aligned import load_common_dataset

            ds = load_common_dataset(year=2024)
            assert ds.n_records == n_tax_units
            assert len(ds.weight) == n_tax_units
            # Check that blind person on tax unit 3 (head) was detected
            assert ds.head_is_blind[3] == True  # noqa: E712  (numpy bool)
            # Check that head who is also a dependent was detected
            assert ds.head_is_dependent[3] == True  # noqa: E712
            # Check that blind spouse on tax unit 3 was detected
            assert ds.spouse_is_blind[3] == True  # noqa: E712


class TestRunAlignedComparison:
    def test_requires_policyengine(self):
        from cosilico_validators.comparison.aligned import HAS_POLICYENGINE

        if not HAS_POLICYENGINE:
            # run_aligned_comparison calls load_common_dataset which needs PE
            from cosilico_validators.comparison.aligned import run_aligned_comparison

            with pytest.raises(ImportError):
                run_aligned_comparison()

    def test_run_with_mock(self):
        """Test run_aligned_comparison with mocked components."""
        import sys

        mock_ds = _make_common_dataset(10)
        mock_pe = MagicMock()
        mock_sim = MagicMock()
        mock_pe.Microsimulation.return_value = mock_sim
        mock_sim.calculate.return_value = np.ones(10) * 500.0

        mock_eitc_func = MagicMock(return_value=np.ones(10) * 500.0)
        mock_tax_func = MagicMock(return_value=np.ones(10) * 3000.0)

        mock_runner = MagicMock()
        mock_runner.calculate_eitc = mock_eitc_func
        mock_runner.calculate_income_tax = mock_tax_func
        mock_runner.PARAMS_2024 = {}

        with (
            patch.dict(
                sys.modules,
                {
                    "policyengine_us": mock_pe,
                    "cosilico_runner": mock_runner,
                    "pandas": MagicMock(),
                },
            ),
            patch("cosilico_validators.comparison.aligned.HAS_POLICYENGINE", True),
            patch("cosilico_validators.comparison.aligned.Microsimulation", mock_pe.Microsimulation),
            patch("cosilico_validators.comparison.aligned.load_common_dataset", return_value=mock_ds),
        ):
            from cosilico_validators.comparison.aligned import run_aligned_comparison

            result = run_aligned_comparison(year=2024)
            assert "metadata" in result
            assert "summary" in result
            assert "variables" in result
