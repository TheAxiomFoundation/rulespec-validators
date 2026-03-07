"""Tests for comparison/taxsim_comparison.py module."""

from unittest.mock import MagicMock, patch

import numpy as np

from cosilico_validators.comparison.taxsim_comparison import (
    ComparisonResult,
    PolicyEngineResult,
    TaxCase,
    TaxSimResult,
    cases_to_taxsim_csv,
    compute_comparison_stats,
    generate_dashboard,
    generate_test_cases,
    main,
    query_taxsim,
    run_comparisons,
    run_policyengine,
)


class TestTaxCase:
    def test_creation(self):
        case = TaxCase(name="test", year=2023, mstat=1, pwages=50000)
        assert case.name == "test"
        assert case.year == 2023
        assert case.mstat == 1
        assert case.pwages == 50000

    def test_defaults(self):
        case = TaxCase(name="test")
        assert case.year == 2023
        assert case.mstat == 1
        assert case.state == 0
        assert case.page == 35
        assert case.depx == 0

    def test_with_dependents(self):
        case = TaxCase(name="test", depx=3, age1=8, age2=10, age3=12)
        assert case.depx == 3
        assert case.age1 == 8


class TestTaxSimResult:
    def test_creation(self):
        result = TaxSimResult(
            taxsim_id=1,
            year=2023,
            state=0,
            fiitax=5000,
            siitax=0,
            fica=3825,
            frate=0.22,
            srate=0,
            ficar=0.0765,
        )
        assert result.taxsim_id == 1
        assert result.fiitax == 5000

    def test_extended_outputs(self):
        result = TaxSimResult(
            taxsim_id=1,
            year=2023,
            state=0,
            fiitax=5000,
            siitax=0,
            fica=3825,
            frate=0.22,
            srate=0,
            ficar=0.0765,
            v25_eitc=600,
            v22_ctc=2000,
        )
        assert result.v25_eitc == 600
        assert result.v22_ctc == 2000


class TestPolicyEngineResult:
    def test_creation(self):
        result = PolicyEngineResult(
            adjusted_gross_income=50000,
            taxable_income=35400,
            income_tax=4000,
            eitc=600,
        )
        assert result.adjusted_gross_income == 50000
        assert result.eitc == 600


class TestComparisonResult:
    def test_creation(self):
        case = TaxCase(name="test")
        taxsim = TaxSimResult(
            taxsim_id=1,
            year=2023,
            state=0,
            fiitax=5000,
            siitax=0,
            fica=3825,
            frate=0.22,
            srate=0,
            ficar=0.0765,
        )
        pe = PolicyEngineResult(income_tax=5000, eitc=600)
        result = ComparisonResult(case=case, taxsim=taxsim, policyengine=pe)
        assert result.case.name == "test"
        assert result.taxsim.fiitax == 5000

    def test_with_errors(self):
        case = TaxCase(name="test")
        result = ComparisonResult(case=case, taxsim=None, policyengine=None, errors=["TAXSIM failed"])
        assert len(result.errors) == 1


class TestGenerateTestCases:
    def test_generates_cases(self):
        cases = generate_test_cases()
        assert len(cases) > 0
        assert all(isinstance(c, TaxCase) for c in cases)

    def test_includes_various_scenarios(self):
        cases = generate_test_cases()
        names = [c.name for c in cases]
        # Should have single filers
        assert any("Single" in n for n in names)
        # Should have married
        assert any("MFJ" in n for n in names)
        # Should have HoH with kids
        assert any("HoH" in n for n in names)
        # Should have self-employed
        assert any("Self-employed" in n for n in names)

    def test_child_ages_set(self):
        cases = generate_test_cases()
        hoh_cases = [c for c in cases if c.depx >= 2]
        assert len(hoh_cases) > 0
        for c in hoh_cases:
            assert c.age1 > 0
            assert c.age2 > 0


class TestCasesToTaxsimCsv:
    def test_csv_generation(self):
        cases = [
            TaxCase(name="test1", year=2023, mstat=1, pwages=50000),
            TaxCase(name="test2", year=2023, mstat=2, pwages=80000, sage=35, swages=20000),
        ]
        csv_str = cases_to_taxsim_csv(cases)
        assert isinstance(csv_str, str)
        lines = csv_str.strip().split("\n")
        assert len(lines) >= 3  # Header + 2 data rows

    def test_csv_includes_header(self):
        cases = [TaxCase(name="test")]
        csv_str = cases_to_taxsim_csv(cases)
        assert "taxsimid" in csv_str
        assert "year" in csv_str

    def test_csv_includes_idtl(self):
        cases = [TaxCase(name="test")]
        csv_str = cases_to_taxsim_csv(cases)
        assert "idtl" in csv_str or "2" in csv_str


class TestComputeComparisonStats:
    def test_with_results(self):
        case = TaxCase(name="test")
        taxsim = TaxSimResult(
            taxsim_id=1,
            year=2023,
            state=0,
            fiitax=5000,
            siitax=0,
            fica=3825,
            frate=0.22,
            srate=0,
            ficar=0.0765,
            v25_eitc=600,
            v22_ctc=2000,
        )
        pe = PolicyEngineResult(income_tax=5000, eitc=600, ctc=2000)
        results = [ComparisonResult(case=case, taxsim=taxsim, policyengine=pe)]
        stats = compute_comparison_stats(results)
        assert isinstance(stats, dict)

    def test_with_errors(self):
        case = TaxCase(name="test")
        results = [ComparisonResult(case=case, taxsim=None, policyengine=None, errors=["failed"])]
        stats = compute_comparison_stats(results)
        assert isinstance(stats, dict)

    def test_with_multiple_comparisons(self):
        """Test with full comparison data to hit all stat branches."""
        cases_data = []
        for i in range(5):
            case = TaxCase(name=f"test_{i}", pwages=(i + 1) * 20000)
            taxsim = TaxSimResult(
                taxsim_id=i + 1,
                year=2023,
                state=0,
                fiitax=1000.0 * (i + 1),
                siitax=0,
                fica=500.0 * (i + 1),
                frate=0.22,
                srate=0,
                ficar=0.0765,
                v10_agi=20000.0 * (i + 1),
                v18_taxable_income=15000.0 * (i + 1),
                v25_eitc=max(0, 600 - i * 200),
                v22_ctc=2000.0 if i < 3 else 0,
                v23_ctc_refundable=500.0 if i < 3 else 0,
                v26_amt=0,
            )
            pe = PolicyEngineResult(
                adjusted_gross_income=20000.0 * (i + 1) + 100,
                taxable_income=15000.0 * (i + 1) + 50,
                income_tax=1000.0 * (i + 1) + 10,
                eitc=max(0, 600 - i * 200 + 5),
                ctc=2000.0 if i < 3 else 0,
                employee_social_security_tax=300.0 * (i + 1),
                self_employment_tax=200.0 * (i + 1),
            )
            cases_data.append(ComparisonResult(case=case, taxsim=taxsim, policyengine=pe))

        stats = compute_comparison_stats(cases_data)
        assert "agi" in stats
        assert "taxable_income" in stats
        assert "federal_tax" in stats
        assert "eitc" in stats
        assert "fica" in stats
        assert stats["agi"]["n"] == 5
        assert stats["agi"]["correlation"] != 0.0
        assert stats["agi"]["pct_exact"] >= 0
        assert stats["agi"]["pct_within_10"] >= 0
        assert stats["agi"]["pct_within_100"] >= 0

    def test_with_amt(self):
        """Test AMT branch in stats."""
        case = TaxCase(name="high_income", pwages=500000)
        taxsim = TaxSimResult(
            taxsim_id=1,
            year=2023,
            state=0,
            fiitax=100000,
            siitax=0,
            fica=20000,
            frate=0.37,
            srate=0,
            ficar=0.0145,
            v10_agi=500000,
            v18_taxable_income=400000,
            v25_eitc=0,
            v22_ctc=0,
            v23_ctc_refundable=0,
            v26_amt=5000,  # AMT > 0 triggers amti tracking
        )
        pe = PolicyEngineResult(
            adjusted_gross_income=500000,
            taxable_income=400000,
            income_tax=100000,
            eitc=0,
            ctc=0,
            employee_social_security_tax=10000,
            self_employment_tax=0,
            amt_income=450000,  # Non-zero AMT income
            amt=5000,
        )
        results = [ComparisonResult(case=case, taxsim=taxsim, policyengine=pe)]
        stats = compute_comparison_stats(results)
        assert "amti" in stats
        assert stats["amti"]["n"] == 1


class TestQueryTaxsim:
    def test_success(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = (
            "taxsimid,year,state,fiitax,siitax,fica,frate,srate,ficar,"
            "v10,v11,v12,v13,v14,v15,v16,v17,v18,v19,v22,v23,v25,v26,v27,v28\n"
            "1,2023,0,5000,0,3825,0.22,0,0.0765,"
            "50000,0,0,0,0,0,14600,0,35400,5000,0,0,600,0,5000,3825"
        )

        with (
            patch("subprocess.run", return_value=mock_result),
            patch("tempfile.NamedTemporaryFile"),
            patch("pathlib.Path.unlink"),
        ):
            results = query_taxsim("csv_data", max_retries=1)
            assert len(results) == 1
            assert results[0].fiitax == 5000
            assert results[0].v25_eitc == 600

    def test_curl_error(self):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "connection refused"

        with (
            patch("subprocess.run", return_value=mock_result),
            patch("tempfile.NamedTemporaryFile"),
            patch("pathlib.Path.unlink"),
        ):
            results = query_taxsim("csv_data", max_retries=1)
            assert results == []

    def test_empty_response(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""

        with (
            patch("subprocess.run", return_value=mock_result),
            patch("tempfile.NamedTemporaryFile"),
            patch("pathlib.Path.unlink"),
        ):
            results = query_taxsim("csv_data", max_retries=1)
            assert results == []

    def test_html_error(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "<html><body>Error page</body></html>"

        with (
            patch("subprocess.run", return_value=mock_result),
            patch("tempfile.NamedTemporaryFile"),
            patch("pathlib.Path.unlink"),
        ):
            results = query_taxsim("csv_data", max_retries=1)
            assert results == []

    def test_timeout(self):
        import subprocess

        with (
            patch("subprocess.run", side_effect=subprocess.TimeoutExpired("curl", 120)),
            patch("tempfile.NamedTemporaryFile"),
            patch("pathlib.Path.unlink"),
            patch("time.sleep"),
        ):
            results = query_taxsim("csv_data", max_retries=2)
            assert results == []

    def test_exception(self):
        with (
            patch("subprocess.run", side_effect=Exception("network error")),
            patch("tempfile.NamedTemporaryFile"),
            patch("pathlib.Path.unlink"),
            patch("time.sleep"),
        ):
            results = query_taxsim("csv_data", max_retries=2)
            assert results == []

    def test_parse_error_in_row(self):
        """Test ValueError/KeyError parsing TAXSIM rows."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        # A row with invalid data that will cause ValueError when float() is called
        mock_result.stdout = (
            "taxsimid,year,state,fiitax,siitax,fica,frate,srate,ficar,"
            "v10,v11,v12,v13,v14,v15,v16,v17,v18,v19,v22,v23,v25,v26,v27,v28\n"
            "bad,bad,bad,bad,bad,bad,bad,bad,bad,"
            "bad,bad,bad,bad,bad,bad,bad,bad,bad,bad,bad,bad,bad,bad,bad,bad"
        )

        with (
            patch("subprocess.run", return_value=mock_result),
            patch("tempfile.NamedTemporaryFile"),
            patch("pathlib.Path.unlink"),
        ):
            results = query_taxsim("csv_data", max_retries=1)
            # Bad rows should be skipped, returning empty results
            assert results == []


class TestRunPolicyengine:
    def test_success(self):
        import sys

        mock_pe = MagicMock()
        mock_sim = MagicMock()
        mock_pe.Simulation.return_value = mock_sim
        mock_sim.calculate.return_value = np.array([500.0])

        case = TaxCase(name="test", pwages=30000)

        with patch.dict(sys.modules, {"policyengine_us": mock_pe}):
            result = run_policyengine(case)
            assert isinstance(result, PolicyEngineResult)

    def test_import_error(self):
        import sys

        # Remove policyengine_us from modules
        saved = sys.modules.get("policyengine_us")
        try:
            sys.modules["policyengine_us"] = None
            case = TaxCase(name="test", pwages=30000)
            result = run_policyengine(case)
            assert isinstance(result, PolicyEngineResult)
            # Defaults should be 0
            assert result.eitc == 0
        finally:
            if saved is not None:
                sys.modules["policyengine_us"] = saved
            else:
                sys.modules.pop("policyengine_us", None)

    def test_with_spouse(self):
        import sys

        mock_pe = MagicMock()
        mock_sim = MagicMock()
        mock_pe.Simulation.return_value = mock_sim
        mock_sim.calculate.return_value = np.array([500.0])

        case = TaxCase(
            name="joint",
            mstat=2,
            pwages=50000,
            sage=38,
            swages=30000,
            ssemp=5000,
        )

        with patch.dict(sys.modules, {"policyengine_us": mock_pe}):
            result = run_policyengine(case)
            assert isinstance(result, PolicyEngineResult)

    def test_with_children(self):
        import sys

        mock_pe = MagicMock()
        mock_sim = MagicMock()
        mock_pe.Simulation.return_value = mock_sim
        mock_sim.calculate.return_value = np.array([500.0])

        case = TaxCase(
            name="hoh",
            mstat=3,
            pwages=30000,
            depx=2,
            age1=8,
            age2=10,
        )

        with patch.dict(sys.modules, {"policyengine_us": mock_pe}):
            result = run_policyengine(case)
            assert isinstance(result, PolicyEngineResult)

    def test_with_itemized_deductions(self):
        import sys

        mock_pe = MagicMock()
        mock_sim = MagicMock()
        mock_pe.Simulation.return_value = mock_sim
        mock_sim.calculate.return_value = np.array([500.0])

        case = TaxCase(
            name="itemized",
            pwages=200000,
            proptax=15000,
            mortgage=20000,
            otheritem=5000,
        )

        with patch.dict(sys.modules, {"policyengine_us": mock_pe}):
            result = run_policyengine(case)
            assert isinstance(result, PolicyEngineResult)

    def test_exception_returns_default(self):
        import sys

        mock_pe = MagicMock()
        mock_pe.Simulation.side_effect = Exception("simulation failed")

        case = TaxCase(name="fail", pwages=30000)

        with patch.dict(sys.modules, {"policyengine_us": mock_pe}):
            result = run_policyengine(case)
            assert isinstance(result, PolicyEngineResult)
            assert result.eitc == 0

    def test_amt_exception(self):
        """Test that AMT exception is caught gracefully."""
        import sys

        mock_pe = MagicMock()
        mock_sim = MagicMock()
        mock_pe.Simulation.return_value = mock_sim
        call_count = [0]

        def mock_calculate(var, year):
            call_count[0] += 1
            if "alternative_minimum" in var:
                raise Exception("AMT not implemented")
            return np.array([500.0])

        mock_sim.calculate.side_effect = mock_calculate
        case = TaxCase(name="test", pwages=30000)

        with patch.dict(sys.modules, {"policyengine_us": mock_pe}):
            result = run_policyengine(case)
            assert isinstance(result, PolicyEngineResult)
            # AMT should remain 0 (default)
            assert result.amt == 0


class TestRunComparisons:
    def test_run_comparisons(self):
        import sys

        cases = [
            TaxCase(name="test1", pwages=30000),
            TaxCase(name="test2", pwages=50000),
        ]

        # Mock TAXSIM results
        ts_results = [
            TaxSimResult(
                taxsim_id=i + 1,
                year=2023,
                state=0,
                fiitax=1000 * (i + 1),
                siitax=0,
                fica=500 * (i + 1),
                frate=0.22,
                srate=0,
                ficar=0.0765,
                v25_eitc=600 - i * 200,
            )
            for i in range(2)
        ]

        mock_pe = MagicMock()
        mock_sim = MagicMock()
        mock_pe.Simulation.return_value = mock_sim
        mock_sim.calculate.return_value = np.array([500.0])

        with (
            patch(
                "cosilico_validators.comparison.taxsim_comparison.query_taxsim",
                return_value=ts_results,
            ),
            patch.dict(sys.modules, {"policyengine_us": mock_pe}),
        ):
            comparisons = run_comparisons(cases)
            assert len(comparisons) == 2
            assert comparisons[0].taxsim is not None
            assert comparisons[0].policyengine is not None

    def test_missing_taxsim_results(self):
        import sys

        cases = [TaxCase(name="test1", pwages=30000)]

        mock_pe = MagicMock()
        mock_sim = MagicMock()
        mock_pe.Simulation.return_value = mock_sim
        mock_sim.calculate.return_value = np.array([500.0])

        with (
            patch(
                "cosilico_validators.comparison.taxsim_comparison.query_taxsim",
                return_value=[],  # No TAXSIM results
            ),
            patch.dict(sys.modules, {"policyengine_us": mock_pe}),
        ):
            comparisons = run_comparisons(cases)
            assert len(comparisons) == 1
            assert comparisons[0].taxsim is None
            assert "TAXSIM result missing" in comparisons[0].errors


class TestGenerateDashboard:
    def test_generate_dashboard(self):
        case = TaxCase(name="test", pwages=50000)
        taxsim = TaxSimResult(
            taxsim_id=1,
            year=2023,
            state=0,
            fiitax=5000,
            siitax=0,
            fica=3825,
            frate=0.22,
            srate=0,
            ficar=0.0765,
            v10_agi=50000,
            v18_taxable_income=35400,
            v25_eitc=0,
            v22_ctc=0,
            v23_ctc_refundable=0,
            v26_amt=0,
            v27_fed_tax_before_credits=5000,
        )
        pe = PolicyEngineResult(
            adjusted_gross_income=50000,
            taxable_income=35400,
            income_tax_before_credits=5000,
            income_tax=5000,
            eitc=0,
            ctc=0,
            employee_social_security_tax=3000,
            self_employment_tax=0,
        )
        comparisons = [ComparisonResult(case=case, taxsim=taxsim, policyengine=pe)]
        stats = compute_comparison_stats(comparisons)
        cases = [case]

        dashboard = generate_dashboard(comparisons, stats, cases)
        assert isinstance(dashboard, str)
        assert "TAXSIM Validation Dashboard" in dashboard
        assert "test" in dashboard
        assert "Summary" in dashboard

    def test_missing_taxsim_in_dashboard(self):
        case = TaxCase(name="test_fail", pwages=50000)
        comparisons = [ComparisonResult(case=case, taxsim=None, policyengine=None)]
        stats = {}
        cases = [case]

        dashboard = generate_dashboard(comparisons, stats, cases)
        assert "TAXSIM Validation Dashboard" in dashboard

    def test_eitc_section(self):
        """Dashboard has EITC-specific section."""
        case = TaxCase(name="eitc_test", pwages=20000, mstat=3, depx=1, age1=8)
        taxsim = TaxSimResult(
            taxsim_id=1,
            year=2023,
            state=0,
            fiitax=0,
            siitax=0,
            fica=1530,
            frate=0.1,
            srate=0,
            ficar=0.0765,
            v25_eitc=3000,
            v22_ctc=2000,
            v23_ctc_refundable=500,
        )
        pe = PolicyEngineResult(eitc=3100, ctc=2100)
        comparisons = [ComparisonResult(case=case, taxsim=taxsim, policyengine=pe)]
        stats = compute_comparison_stats(comparisons)

        dashboard = generate_dashboard(comparisons, stats, [case])
        assert "EITC Discrepancies" in dashboard
        assert "CTC Discrepancies" in dashboard

    def test_largest_discrepancies(self):
        """Dashboard shows largest discrepancies section."""
        cases = []
        comparisons = []
        for i in range(5):
            case = TaxCase(name=f"case_{i}", pwages=(i + 1) * 30000)
            taxsim = TaxSimResult(
                taxsim_id=i + 1,
                year=2023,
                state=0,
                fiitax=1000 * (i + 1),
                siitax=0,
                fica=500 * (i + 1),
                frate=0.22,
                srate=0,
                ficar=0.0765,
                v10_agi=30000 * (i + 1),
            )
            pe = PolicyEngineResult(
                income_tax=1000 * (i + 1) + 100 * (i + 1),
                adjusted_gross_income=30000 * (i + 1),
            )
            cases.append(case)
            comparisons.append(ComparisonResult(case=case, taxsim=taxsim, policyengine=pe))

        stats = compute_comparison_stats(comparisons)
        dashboard = generate_dashboard(comparisons, stats, cases)
        assert "Largest Discrepancies" in dashboard


class TestMainFunction:
    def test_main(self, tmp_path):
        """main() generates test cases, runs comparisons, writes output."""
        import sys

        mock_pe = MagicMock()
        mock_sim = MagicMock()
        mock_pe.Simulation.return_value = mock_sim
        mock_sim.calculate.return_value = np.array([500.0])

        # Return enough TAXSIM results to cover the summary print loop (line 996)
        ts_results = []
        for i in range(len(generate_test_cases())):
            ts_results.append(
                TaxSimResult(
                    taxsim_id=i + 1,
                    year=2023,
                    state=0,
                    fiitax=5000.0,
                    siitax=0,
                    fica=3825,
                    frate=0.22,
                    srate=0,
                    ficar=0.0765,
                    v10_agi=50000,
                    v18_taxable_income=35400,
                    v25_eitc=600,
                    v22_ctc=2000,
                    v23_ctc_refundable=500,
                    v26_amt=0,
                )
            )

        with (
            patch(
                "cosilico_validators.comparison.taxsim_comparison.query_taxsim",
                return_value=ts_results,
            ),
            patch.dict(sys.modules, {"policyengine_us": mock_pe}),
            patch("pathlib.Path.write_text"),
            patch("pathlib.Path.mkdir"),
        ):
            main()

    def test_main_empty_taxsim(self, tmp_path):
        """main() with empty TAXSIM results."""
        import sys

        mock_pe = MagicMock()
        mock_sim = MagicMock()
        mock_pe.Simulation.return_value = mock_sim
        mock_sim.calculate.return_value = np.array([500.0])

        with (
            patch(
                "cosilico_validators.comparison.taxsim_comparison.query_taxsim",
                return_value=[],
            ),
            patch.dict(sys.modules, {"policyengine_us": mock_pe}),
            patch("pathlib.Path.write_text"),
            patch("pathlib.Path.mkdir"),
        ):
            main()
