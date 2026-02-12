"""Tests for dashboard_export module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cosilico_validators.comparison.aligned import ComparisonResult
from cosilico_validators.dashboard_export import (
    CDCC_PARAMS_2024,
    CTC_PARAMS_2024,
    EITC_PARAMS_2024,
    NIIT_PARAMS_2024,
    STD_DEDUCTION_PARAMS_2024,
    VARIABLES,
    get_git_commit,
    load_cosilico_engine,
    load_rac_file,
    main,
    result_to_section,
    run_export,
)


class TestGetGitCommit:
    def test_returns_string(self):
        result = get_git_commit()
        assert isinstance(result, str)

    def test_returns_unknown_on_failure(self):
        with patch("subprocess.run", side_effect=Exception("fail")):
            assert get_git_commit() == "unknown"

    def test_returns_unknown_on_empty(self):
        mock = MagicMock()
        mock.stdout = ""
        with patch("subprocess.run", return_value=mock):
            assert get_git_commit() == "unknown"


class TestLoadRacFile:
    def test_load_nonexistent(self):
        result = load_rac_file("nonexistent/section")
        assert result is None or isinstance(result, str)

    def test_load_direct_path(self, tmp_path):
        # Create a fake statute directory structure
        statute_dir = tmp_path / "CosilicoAI" / "cosilico-us" / "statute"
        rac_dir = statute_dir / "26"
        rac_dir.mkdir(parents=True)
        (rac_dir / "32.rac").write_text("test content")

        with patch("cosilico_validators.dashboard_export.Path.home", return_value=tmp_path):
            result = load_rac_file("26/32")
            assert result == "test content"

    def test_load_subdirectory_path(self, tmp_path):
        statute_dir = tmp_path / "CosilicoAI" / "cosilico-us" / "statute"
        rac_dir = statute_dir / "26" / "63"
        rac_dir.mkdir(parents=True)
        (rac_dir / "a.rac").write_text("sub content")

        with patch("cosilico_validators.dashboard_export.Path.home", return_value=tmp_path):
            result = load_rac_file("26/63")
            assert result == "sub content"

    def test_load_not_found(self, tmp_path):
        statute_dir = tmp_path / "CosilicoAI" / "cosilico-us" / "statute"
        statute_dir.mkdir(parents=True)

        with patch("cosilico_validators.dashboard_export.Path.home", return_value=tmp_path):
            result = load_rac_file("99/99")
            assert result is None


class TestResultToSection:
    def _make_result(self, variable="eitc", match_rate=0.95, mae=50.0,
                     cos_total=60e9, pe_total=62e9, n=100):
        return ComparisonResult(
            variable=variable,
            match_rate=match_rate,
            mean_absolute_error=mae,
            n_records=n,
            cosilico_total=cos_total,
            policyengine_total=pe_total,
            cosilico_values=np.zeros(n),
            policyengine_values=np.zeros(n),
            error_percentiles={"p50": 10, "p90": 50, "p95": 100, "p99": 500, "max": 1000},
        )

    def test_implemented_variable(self):
        result = self._make_result()
        meta = {"section": "26/32", "title": "EITC"}
        section = result_to_section(result, 100000, meta, implemented=True)
        assert section["section"] == "26/32"
        assert section["variable"] == "eitc"
        assert section["implemented"] is True
        assert section["summary"]["matchRate"] == 0.95
        assert "Cosilico total" in section["notes"]

    def test_unimplemented_variable(self):
        result = self._make_result(variable="ctc", match_rate=0.0, mae=0.0,
                                   cos_total=0.0, pe_total=50e9)
        meta = {"section": "26/24", "title": "CTC"}
        section = result_to_section(result, 100000, meta, implemented=False)
        assert section["implemented"] is False
        assert "Not yet implemented" in section["notes"]

    def test_section_structure(self):
        result = self._make_result()
        meta = {"section": "26/32", "title": "EITC"}
        section = result_to_section(result, 100000, meta, implemented=True)
        assert "households" in section
        assert "testCases" in section
        assert "summary" in section
        assert "validatorBreakdown" in section
        assert section["households"] == 100000


class TestParams:
    def test_eitc_params(self):
        assert EITC_PARAMS_2024["credit_rate_0"] == 0.0765
        assert "earned_income_cap_1" in EITC_PARAMS_2024

    def test_niit_params(self):
        assert NIIT_PARAMS_2024["niit_rate"] == 0.038
        assert NIIT_PARAMS_2024["threshold_joint"] == 250000

    def test_ctc_params(self):
        assert CTC_PARAMS_2024["credit_amount"] == 2000
        assert CTC_PARAMS_2024["refundable_max"] == 1700

    def test_cdcc_params(self):
        assert CDCC_PARAMS_2024["maximum_rate"] == 0.35
        assert CDCC_PARAMS_2024["minimum_rate"] == 0.20

    def test_std_deduction_params(self):
        assert STD_DEDUCTION_PARAMS_2024["basic_single"] == 14600
        assert STD_DEDUCTION_PARAMS_2024["basic_joint"] == 29200


class TestVariables:
    def test_variables_dict(self):
        assert "eitc" in VARIABLES
        assert VARIABLES["eitc"]["section"] == "26/32"
        assert "title" in VARIABLES["eitc"]

    def test_all_variables_have_section_and_title(self):
        for name, meta in VARIABLES.items():
            assert "section" in meta, f"{name} missing section"
            assert "title" in meta, f"{name} missing title"


class TestLoadCosilicoEngine:
    def test_import_error(self):
        with patch("cosilico_validators.dashboard_export.Path.home") as mock_home:
            mock_home.return_value = Path("/nonexistent")
            with pytest.raises(ImportError):
                load_cosilico_engine()

    def test_import_success(self):
        mock_ve = MagicMock()
        mock_parser = MagicMock()
        mock_dr = MagicMock()
        with patch("cosilico_validators.dashboard_export.Path.home") as mock_home, \
             patch.dict("sys.modules", {
                "cosilico": MagicMock(),
                "cosilico.vectorized_executor": MagicMock(VectorizedExecutor=mock_ve),
                "cosilico.dsl_parser": MagicMock(parse_dsl=mock_parser),
                "cosilico.dependency_resolver": MagicMock(DependencyResolver=mock_dr),
             }):
            mock_home.return_value = Path("/nonexistent")
            ve, parser, dr = load_cosilico_engine()
            assert ve is mock_ve
            assert parser is mock_parser
            assert dr is mock_dr


class TestRunExport:
    def test_run_export_requires_policyengine(self):
        with pytest.raises(ImportError), patch.dict("sys.modules", {"policyengine_us": None}):
            run_export(year=2024)

    def test_run_export_full_mock(self, tmp_path):
        """Test run_export with all dependencies mocked."""
        n = 100

        # Create a mock CommonDataset
        mock_dataset = MagicMock()
        mock_dataset.n_records = n
        mock_dataset.is_joint = np.zeros(n, dtype=bool)
        mock_dataset.eitc_child_count = np.zeros(n)
        mock_dataset.earned_income = np.ones(n) * 30000
        mock_dataset.adjusted_gross_income = np.ones(n) * 30000
        mock_dataset.wages = np.ones(n) * 30000
        mock_dataset.self_employment_income = np.zeros(n)
        mock_dataset.partnership_s_corp_income = np.zeros(n)
        mock_dataset.farm_income = np.zeros(n)
        mock_dataset.interest_income = np.zeros(n)
        mock_dataset.dividend_income = np.zeros(n)
        mock_dataset.capital_gains = np.zeros(n)
        mock_dataset.rental_income = np.zeros(n)
        mock_dataset.taxable_social_security = np.zeros(n)
        mock_dataset.pension_income = np.zeros(n)
        mock_dataset.taxable_unemployment = np.zeros(n)
        mock_dataset.retirement_distributions = np.zeros(n)
        mock_dataset.miscellaneous_income = np.zeros(n)
        mock_dataset.investment_income = np.zeros(n)
        mock_dataset.ctc_child_count = np.zeros(n)
        mock_dataset.head_age = np.ones(n) * 30
        mock_dataset.head_is_blind = np.zeros(n, dtype=bool)
        mock_dataset.spouse_is_blind = np.zeros(n, dtype=bool)
        mock_dataset.head_is_dependent = np.zeros(n, dtype=bool)
        mock_dataset.cdcc_qualifying_individuals = np.zeros(n)
        mock_dataset.childcare_expenses = np.zeros(n)
        mock_dataset.self_employment_tax_deduction = np.zeros(n)
        mock_dataset.self_employed_health_insurance_deduction = np.zeros(n)
        mock_dataset.educator_expense_deduction = np.zeros(n)
        mock_dataset.loss_deduction = np.zeros(n)
        mock_dataset.self_employed_pension_deduction = np.zeros(n)
        mock_dataset.ira_deduction = np.zeros(n)
        mock_dataset.hsa_deduction = np.zeros(n)
        mock_dataset.student_loan_interest_deduction = np.zeros(n)
        mock_dataset.above_the_line_deductions_total = np.zeros(n)

        # Mock compare_variable to return a ComparisonResult
        mock_result = ComparisonResult(
            variable="eitc",
            match_rate=0.95,
            mean_absolute_error=50.0,
            n_records=n,
            cosilico_total=60e9,
            policyengine_total=62e9,
            cosilico_values=np.zeros(n),
            policyengine_values=np.zeros(n),
            error_percentiles={"p50": 10, "p90": 50, "p95": 100, "p99": 500, "max": 1000},
        )

        # Mock Microsimulation
        mock_microsim = MagicMock()
        mock_microsim_cls = MagicMock(return_value=mock_microsim)
        mock_microsim.calculate.return_value = np.zeros(n)

        # Mock policyengine_us module
        mock_pe_module = MagicMock()
        mock_pe_module.Microsimulation = mock_microsim_cls

        output_path = tmp_path / "output.json"

        with patch.dict("sys.modules", {"policyengine_us": mock_pe_module}), \
             patch("cosilico_validators.dashboard_export.load_common_dataset", return_value=mock_dataset), \
             patch("cosilico_validators.dashboard_export.compare_variable", return_value=mock_result), \
             patch("cosilico_validators.dashboard_export.load_cosilico_engine", side_effect=ImportError("no engine")), \
             patch("cosilico_validators.dashboard_export.load_rac_file", return_value=None):
            data = run_export(year=2024, output_path=output_path)

            assert "sections" in data
            assert "coverage" in data
            assert "overall" in data
            assert "validators" in data
            assert data["householdsTotal"] == n
            assert output_path.exists()

    def test_run_export_with_engine(self, tmp_path):
        """Test run_export when engine IS available."""
        n = 100

        mock_dataset = MagicMock()
        mock_dataset.n_records = n
        mock_dataset.is_joint = np.zeros(n, dtype=bool)
        mock_dataset.eitc_child_count = np.zeros(n)
        mock_dataset.earned_income = np.ones(n) * 30000
        mock_dataset.adjusted_gross_income = np.ones(n) * 30000
        mock_dataset.wages = np.ones(n) * 30000
        mock_dataset.self_employment_income = np.zeros(n)
        mock_dataset.partnership_s_corp_income = np.zeros(n)
        mock_dataset.farm_income = np.zeros(n)
        mock_dataset.interest_income = np.zeros(n)
        mock_dataset.dividend_income = np.zeros(n)
        mock_dataset.capital_gains = np.zeros(n)
        mock_dataset.rental_income = np.zeros(n)
        mock_dataset.taxable_social_security = np.zeros(n)
        mock_dataset.pension_income = np.zeros(n)
        mock_dataset.taxable_unemployment = np.zeros(n)
        mock_dataset.retirement_distributions = np.zeros(n)
        mock_dataset.miscellaneous_income = np.zeros(n)
        mock_dataset.investment_income = np.zeros(n)
        mock_dataset.ctc_child_count = np.zeros(n)
        mock_dataset.head_age = np.ones(n) * 30
        mock_dataset.head_is_blind = np.zeros(n, dtype=bool)
        mock_dataset.spouse_is_blind = np.zeros(n, dtype=bool)
        mock_dataset.head_is_dependent = np.zeros(n, dtype=bool)
        mock_dataset.cdcc_qualifying_individuals = np.zeros(n)
        mock_dataset.childcare_expenses = np.zeros(n)
        mock_dataset.self_employment_tax_deduction = np.zeros(n)
        mock_dataset.self_employed_health_insurance_deduction = np.zeros(n)
        mock_dataset.educator_expense_deduction = np.zeros(n)
        mock_dataset.loss_deduction = np.zeros(n)
        mock_dataset.self_employed_pension_deduction = np.zeros(n)
        mock_dataset.ira_deduction = np.zeros(n)
        mock_dataset.hsa_deduction = np.zeros(n)
        mock_dataset.student_loan_interest_deduction = np.zeros(n)
        mock_dataset.above_the_line_deductions_total = np.zeros(n)

        mock_result = ComparisonResult(
            variable="eitc",
            match_rate=0.95,
            mean_absolute_error=50.0,
            n_records=n,
            cosilico_total=60e9,
            policyengine_total=62e9,
            cosilico_values=np.zeros(n),
            policyengine_values=np.zeros(n),
            error_percentiles={"p50": 10, "p90": 50, "p95": 100, "p99": 500, "max": 1000},
        )

        mock_microsim = MagicMock()
        mock_microsim_cls = MagicMock(return_value=mock_microsim)
        mock_microsim.calculate.return_value = np.zeros(n)
        mock_pe_module = MagicMock()
        mock_pe_module.Microsimulation = mock_microsim_cls

        mock_executor = MagicMock()
        mock_executor.execute.return_value = {"eitc_standalone": np.ones(n) * 500}
        mock_executor.execute_lazy.return_value = {"adjusted_gross_income": np.ones(n) * 30000}
        mock_executor_cls = MagicMock(return_value=mock_executor)
        mock_parser = MagicMock()
        mock_dep_resolver_cls = MagicMock()

        with patch.dict("sys.modules", {"policyengine_us": mock_pe_module}), \
             patch("cosilico_validators.dashboard_export.load_common_dataset", return_value=mock_dataset), \
             patch("cosilico_validators.dashboard_export.compare_variable", return_value=mock_result), \
             patch("cosilico_validators.dashboard_export.load_cosilico_engine",
                    return_value=(mock_executor_cls, mock_parser, mock_dep_resolver_cls)), \
             patch("cosilico_validators.dashboard_export.load_rac_file", return_value="mock rac code"), \
             patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.read_text", return_value="mock rac code"):
            data = run_export(year=2024)

            assert "sections" in data
            assert len(data["sections"]) > 0
            assert data["coverage"]["total"] > 0

    def test_run_export_variable_error(self, tmp_path):
        """Test run_export when a variable comparison raises an exception."""
        n = 100

        mock_dataset = MagicMock()
        mock_dataset.n_records = n

        mock_microsim = MagicMock()
        mock_microsim_cls = MagicMock(return_value=mock_microsim)
        mock_microsim.calculate.side_effect = Exception("PE calculation failed")
        mock_pe_module = MagicMock()
        mock_pe_module.Microsimulation = mock_microsim_cls

        with patch.dict("sys.modules", {"policyengine_us": mock_pe_module}), \
             patch("cosilico_validators.dashboard_export.load_common_dataset", return_value=mock_dataset), \
             patch("cosilico_validators.dashboard_export.load_cosilico_engine", side_effect=ImportError("no engine")):
            data = run_export(year=2024)
            # Should still return valid structure even with errors
            assert "sections" in data
            assert "coverage" in data

    def test_run_export_no_output_path(self):
        """Test run_export without output path - should not write file."""
        n = 10
        mock_dataset = MagicMock()
        mock_dataset.n_records = n

        mock_result = ComparisonResult(
            variable="eitc",
            match_rate=0.5,
            mean_absolute_error=100.0,
            n_records=n,
            cosilico_total=0,
            policyengine_total=62e9,
            cosilico_values=np.zeros(n),
            policyengine_values=np.zeros(n),
            error_percentiles={"p50": 10, "p90": 50, "p95": 100, "p99": 500, "max": 1000},
        )

        mock_microsim = MagicMock()
        mock_microsim_cls = MagicMock(return_value=mock_microsim)
        mock_microsim.calculate.return_value = np.zeros(n)
        mock_pe_module = MagicMock()
        mock_pe_module.Microsimulation = mock_microsim_cls

        with patch.dict("sys.modules", {"policyengine_us": mock_pe_module}), \
             patch("cosilico_validators.dashboard_export.load_common_dataset", return_value=mock_dataset), \
             patch("cosilico_validators.dashboard_export.compare_variable", return_value=mock_result), \
             patch("cosilico_validators.dashboard_export.load_cosilico_engine", side_effect=ImportError("no engine")), \
             patch("cosilico_validators.dashboard_export.load_rac_file", return_value=None):
            data = run_export(year=2024)
            assert isinstance(data, dict)


class TestRunExportEngineBranches:
    """Test specific engine integration branches within run_export."""

    def _make_mock_dataset(self, n=10):
        ds = MagicMock()
        ds.n_records = n
        ds.is_joint = np.zeros(n, dtype=bool)
        ds.eitc_child_count = np.zeros(n)
        ds.ctc_child_count = np.zeros(n)
        ds.earned_income = np.ones(n) * 30000
        ds.adjusted_gross_income = np.ones(n) * 30000
        ds.taxable_income = np.ones(n) * 20000
        ds.wages = np.ones(n) * 30000
        ds.self_employment_income = np.zeros(n)
        ds.partnership_s_corp_income = np.zeros(n)
        ds.farm_income = np.zeros(n)
        ds.interest_income = np.zeros(n)
        ds.dividend_income = np.zeros(n)
        ds.capital_gains = np.zeros(n)
        ds.rental_income = np.zeros(n)
        ds.taxable_social_security = np.zeros(n)
        ds.pension_income = np.zeros(n)
        ds.taxable_unemployment = np.zeros(n)
        ds.retirement_distributions = np.zeros(n)
        ds.miscellaneous_income = np.zeros(n)
        ds.investment_income = np.zeros(n)
        ds.head_age = np.ones(n) * 35
        ds.spouse_age = np.zeros(n)
        ds.head_is_blind = np.zeros(n, dtype=bool)
        ds.spouse_is_blind = np.zeros(n, dtype=bool)
        ds.head_is_dependent = np.zeros(n, dtype=bool)
        ds.cdcc_qualifying_individuals = np.zeros(n)
        ds.childcare_expenses = np.zeros(n)
        ds.self_employment_tax_deduction = np.zeros(n)
        ds.self_employed_health_insurance_deduction = np.zeros(n)
        ds.educator_expense_deduction = np.zeros(n)
        ds.loss_deduction = np.zeros(n)
        ds.self_employed_pension_deduction = np.zeros(n)
        ds.ira_deduction = np.zeros(n)
        ds.hsa_deduction = np.zeros(n)
        ds.student_loan_interest_deduction = np.zeros(n)
        ds.above_the_line_deductions_total = np.zeros(n)
        return ds

    def _run_with_engine(self, n=10, engine_exception=None):
        """Helper to run export with engine available, exercising all branches."""
        mock_dataset = self._make_mock_dataset(n)

        mock_result = ComparisonResult(
            variable="eitc", match_rate=0.95, mean_absolute_error=50.0,
            n_records=n, cosilico_total=60e9, policyengine_total=62e9,
            cosilico_values=np.zeros(n), policyengine_values=np.zeros(n),
            error_percentiles={"p50": 10, "p90": 50, "p95": 100, "p99": 500, "max": 1000},
        )

        mock_microsim = MagicMock()
        mock_microsim_cls = MagicMock(return_value=mock_microsim)
        mock_microsim.calculate.return_value = np.zeros(n)
        mock_pe_module = MagicMock()
        mock_pe_module.Microsimulation = mock_microsim_cls

        mock_executor = MagicMock()
        if engine_exception:
            mock_executor.execute.side_effect = engine_exception
            mock_executor.execute_lazy.side_effect = engine_exception
        else:
            mock_executor.execute.return_value = {
                "eitc_standalone": np.ones(n) * 500,
                "niit_standalone": np.zeros(n),
                "cdcc_standalone": np.zeros(n),
                "standard_deduction_standalone": np.ones(n) * 14600,
            }
            mock_executor.execute_lazy.return_value = {
                "adjusted_gross_income": np.ones(n) * 30000,
                "ctc_total": np.ones(n) * 2000,
                "child_tax_credit": np.ones(n) * 2000,
                "additional_child_tax_credit": np.zeros(n),
            }
        mock_executor_cls = MagicMock(return_value=mock_executor)
        mock_dep_resolver_cls = MagicMock()

        mock_rac_path = MagicMock()
        mock_rac_path.exists.return_value = True
        mock_rac_path.read_text.return_value = "mock rac code"

        with patch.dict("sys.modules", {"policyengine_us": mock_pe_module}), \
             patch("cosilico_validators.dashboard_export.load_common_dataset", return_value=mock_dataset), \
             patch("cosilico_validators.dashboard_export.compare_variable", return_value=mock_result), \
             patch("cosilico_validators.dashboard_export.load_cosilico_engine",
                    return_value=(mock_executor_cls, MagicMock(), mock_dep_resolver_cls)), \
             patch("cosilico_validators.dashboard_export.load_rac_file", return_value="mock rac"), \
             patch("cosilico_validators.dashboard_export.Path") as mock_path_cls:
            # Make Path.home() / ... / "statute" / "26" / "32.rac" all return mock_rac_path
            mock_path_cls.home.return_value.__truediv__ = MagicMock(return_value=mock_rac_path)
            mock_path_cls.__truediv__ = MagicMock(return_value=mock_rac_path)
            mock_rac_path.__truediv__ = MagicMock(return_value=mock_rac_path)

            data = run_export(year=2024)
            return data

    def test_engine_all_branches_success(self):
        """Exercise all engine integration branches successfully."""
        data = self._run_with_engine()
        assert "sections" in data
        assert len(data["sections"]) > 0

    def test_engine_all_branches_failure(self):
        """Exercise all engine integration branches with engine exceptions."""
        data = self._run_with_engine(engine_exception=Exception("engine failed"))
        assert "sections" in data
        # Variables should still be in results (as unimplemented)
        assert len(data["sections"]) > 0

    def test_load_engine_path_exists(self, tmp_path):
        """Test load_cosilico_engine when path exists (line 91)."""
        engine_dir = tmp_path / "CosilicoAI" / "cosilico-engine" / "src"
        engine_dir.mkdir(parents=True)

        mock_ve = MagicMock()
        mock_parser = MagicMock()
        mock_dr = MagicMock()
        with patch("cosilico_validators.dashboard_export.Path.home", return_value=tmp_path), \
             patch.dict("sys.modules", {
                "cosilico": MagicMock(),
                "cosilico.vectorized_executor": MagicMock(VectorizedExecutor=mock_ve),
                "cosilico.dsl_parser": MagicMock(parse_dsl=mock_parser),
                "cosilico.dependency_resolver": MagicMock(DependencyResolver=mock_dr),
             }):
            ve, parser, dr = load_cosilico_engine()
            assert ve is mock_ve


class TestMainCommand:
    def test_main_invokable(self):
        from click.testing import CliRunner
        runner = CliRunner()
        with patch("cosilico_validators.dashboard_export.run_export") as mock_run:
            mock_run.return_value = {
                "coverage": {"implemented": 5, "total": 13},
                "overall": {"matchRate": 0.95, "meanAbsoluteError": 50},
            }
            result = runner.invoke(main, [])
            assert result.exit_code == 0

    def test_main_with_output(self, tmp_path):
        from click.testing import CliRunner
        runner = CliRunner()
        output_path = str(tmp_path / "output.json")
        with patch("cosilico_validators.dashboard_export.run_export") as mock_run:
            mock_run.return_value = {
                "coverage": {"implemented": 5, "total": 13},
                "overall": {"matchRate": 0.95, "meanAbsoluteError": 50},
            }
            result = runner.invoke(main, ["-o", output_path])
            assert result.exit_code == 0
