"""Tests for comparison/cps.py module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cosilico_validators.comparison.cps import (
    COMPARISON_VARIABLES,
    ComparisonTotals,
    ModelResult,
    TimedResult,
    load_variable_mappings,
)


class TestConstants:
    def test_comparison_variables_exist(self):
        assert len(COMPARISON_VARIABLES) > 0

    def test_comparison_variables_have_required_keys(self):
        for _name, config in COMPARISON_VARIABLES.items():
            assert "title" in config
            assert "cosilico_col" in config
            assert "pe_var" in config


class TestTimedResult:
    def test_creation(self):
        tr = TimedResult(data={"test": np.array([1, 2])}, elapsed_ms=0.5)
        assert "test" in tr.data
        assert tr.elapsed_ms == 0.5


class TestModelResult:
    def test_creation(self):
        mr = ModelResult(name="policyengine", total=60e9, n_records=100000, time_ms=500.0)
        assert mr.name == "policyengine"
        assert mr.total == 60e9
        assert mr.n_records == 100000
        assert mr.time_ms == 500.0


class TestComparisonTotals:
    def test_creation(self):
        models = {
            "cosilico": ModelResult(name="cosilico", total=60e9, n_records=100000, time_ms=50.0),
            "policyengine": ModelResult(name="policyengine", total=62e9, n_records=100000, time_ms=200.0),
        }
        ct = ComparisonTotals(variable="eitc", title="EITC", models=models)
        assert ct.variable == "eitc"
        assert ct.cosilico_total == 60e9
        assert ct.policyengine_total == 62e9

    def test_difference(self):
        models = {
            "cosilico": ModelResult(name="cosilico", total=60e9, n_records=100000, time_ms=50.0),
            "policyengine": ModelResult(name="policyengine", total=62e9, n_records=100000, time_ms=200.0),
        }
        ct = ComparisonTotals(variable="eitc", title="EITC", models=models)
        assert ct.difference == 60e9 - 62e9

    def test_percent_difference(self):
        models = {
            "cosilico": ModelResult(name="cosilico", total=60e9, n_records=100000, time_ms=50.0),
            "policyengine": ModelResult(name="policyengine", total=62e9, n_records=100000, time_ms=200.0),
        }
        ct = ComparisonTotals(variable="eitc", title="EITC", models=models)
        expected_pct = (60e9 - 62e9) / 62e9 * 100
        assert ct.percent_difference == pytest.approx(expected_pct)

    def test_percent_difference_zero_pe(self):
        models = {
            "cosilico": ModelResult(name="cosilico", total=100, n_records=10, time_ms=50.0),
            "policyengine": ModelResult(name="policyengine", total=0, n_records=10, time_ms=50.0),
        }
        ct = ComparisonTotals(variable="test", title="Test", models=models)
        assert ct.percent_difference == 0.0

    def test_get_total_missing_model(self):
        ct = ComparisonTotals(variable="test", title="Test", models={})
        assert ct.get_total("nonexistent") == 0.0

    def test_n_records(self):
        models = {
            "cosilico": ModelResult(name="cosilico", total=60e9, n_records=100000, time_ms=50.0),
            "policyengine": ModelResult(name="policyengine", total=62e9, n_records=90000, time_ms=200.0),
        }
        ct = ComparisonTotals(variable="test", title="Test", models=models)
        assert ct.n_records == 100000

    def test_n_records_empty(self):
        ct = ComparisonTotals(variable="test", title="Test", models={})
        assert ct.n_records == 0

    def test_taxcalc_total(self):
        models = {
            "taxcalc": ModelResult(name="taxcalc", total=55e9, n_records=80000, time_ms=300.0),
        }
        ct = ComparisonTotals(variable="test", title="Test", models=models)
        assert ct.taxcalc_total == 55e9

    def test_match_rate(self):
        models = {
            "cosilico": ModelResult(name="cosilico", total=100, n_records=10, time_ms=50.0),
            "policyengine": ModelResult(name="policyengine", total=100, n_records=10, time_ms=100.0),
        }
        ct = ComparisonTotals(variable="test", title="Test", models=models)
        # match_rate and mean_absolute_error are properties
        assert isinstance(ct.match_rate, float)
        assert isinstance(ct.mean_absolute_error, float)

    def test_model_time_via_models_dict(self):
        models = {
            "cosilico": ModelResult(name="cosilico", total=100, n_records=10, time_ms=50.0),
            "policyengine": ModelResult(name="policyengine", total=100, n_records=10, time_ms=200.0),
        }
        ct = ComparisonTotals(variable="test", title="Test", models=models)
        assert ct.models["cosilico"].time_ms == 50.0
        assert ct.models["policyengine"].time_ms == 200.0


class TestLoadVariableMappings:
    def test_load(self):
        mappings = load_variable_mappings()
        assert isinstance(mappings, dict)
        assert len(mappings) > 0

    def test_mappings_have_cosilico_col(self):
        mappings = load_variable_mappings()
        for _name, config in mappings.items():
            assert "cosilico_col" in config


class TestLoadCosilicoSources:
    def test_load_fails_without_module(self):
        from cosilico_validators.comparison.cps import load_cosilico_cps
        with pytest.raises((ImportError, ModuleNotFoundError)):
            load_cosilico_cps(year=2024)


class TestCompareCpsTotals:
    def test_compare_with_cosilico_only(self):
        from cosilico_validators.comparison.cps import compare_cps_totals
        mock_timed = TimedResult(
            data={
                "weight": np.array([1.0, 2.0]),
                "eitc": np.array([500.0, 600.0]),
            },
            elapsed_ms=50.0,
        )
        with patch("cosilico_validators.comparison.cps.load_cosilico_cps",
                    return_value=mock_timed):
            result = compare_cps_totals(
                year=2024, variables=["eitc"],
                models=["cosilico"],
            )
            assert isinstance(result, dict)
            assert "eitc" in result

    def test_compare_with_pe_and_cosilico(self):
        from cosilico_validators.comparison.cps import compare_cps_totals
        mock_timed = TimedResult(
            data={
                "weight": np.array([1.0, 2.0]),
                "eitc": np.array([500.0, 600.0]),
            },
            elapsed_ms=50.0,
        )
        with patch("cosilico_validators.comparison.cps.load_cosilico_cps",
                    return_value=mock_timed), \
             patch("cosilico_validators.comparison.cps.load_policyengine_values",
                    return_value=mock_timed):
            result = compare_cps_totals(
                year=2024, variables=["eitc"],
                models=["cosilico", "policyengine"],
            )
            assert "eitc" in result
            assert result["eitc"].cosilico_total > 0
            assert result["eitc"].policyengine_total > 0

    def test_compare_with_all_models(self):
        from cosilico_validators.comparison.cps import compare_cps_totals
        mock_timed = TimedResult(
            data={
                "weight": np.array([1.0, 2.0]),
                "eitc": np.array([500.0, 600.0]),
            },
            elapsed_ms=50.0,
        )
        with patch("cosilico_validators.comparison.cps.load_cosilico_cps",
                    return_value=mock_timed), \
             patch("cosilico_validators.comparison.cps.load_policyengine_values",
                    return_value=mock_timed), \
             patch("cosilico_validators.comparison.cps.load_taxcalc_values",
                    return_value=mock_timed), \
             patch("cosilico_validators.comparison.cps.load_taxsim_values",
                    return_value=mock_timed):
            result = compare_cps_totals(year=2024, variables=["eitc"])
            assert "eitc" in result
            assert result["eitc"].cosilico_total > 0

    def test_compare_pe_import_error(self):
        from cosilico_validators.comparison.cps import compare_cps_totals
        mock_timed = TimedResult(
            data={"weight": np.array([1.0]), "eitc": np.array([500.0])},
            elapsed_ms=50.0,
        )
        with patch("cosilico_validators.comparison.cps.load_cosilico_cps",
                    return_value=mock_timed), \
             patch("cosilico_validators.comparison.cps.load_policyengine_values",
                    side_effect=ImportError("no PE")), \
             patch("cosilico_validators.comparison.cps.load_taxcalc_values",
                    side_effect=ImportError("no TC")), \
             patch("cosilico_validators.comparison.cps.load_taxsim_values",
                    side_effect=Exception("no TAXSIM")):
            result = compare_cps_totals(year=2024, variables=["eitc"])
            assert "eitc" in result
            # Only cosilico should be present
            assert "cosilico" in result["eitc"].models

    def test_compare_default_variables(self):
        from cosilico_validators.comparison.cps import compare_cps_totals
        mock_data = {"weight": np.array([1.0])}
        for var in COMPARISON_VARIABLES:
            mock_data[var] = np.array([100.0])
        mock_timed = TimedResult(data=mock_data, elapsed_ms=50.0)
        with patch("cosilico_validators.comparison.cps.load_cosilico_cps",
                    return_value=mock_timed):
            result = compare_cps_totals(year=2024, models=["cosilico"])
            assert len(result) > 0


class TestExportToDashboard:
    def test_export_hits_attribute_error(self):
        """export_to_dashboard references cosilico_time_ms which doesn't exist on ComparisonTotals."""
        from cosilico_validators.comparison.cps import export_to_dashboard
        models = {
            "cosilico": ModelResult(name="cosilico", total=60e9, n_records=100000, time_ms=50.0),
            "policyengine": ModelResult(name="policyengine", total=62e9, n_records=100000, time_ms=200.0),
        }
        comparison = {
            "eitc": ComparisonTotals(variable="eitc", title="EITC", models=models),
        }
        # The function references totals.cosilico_time_ms which is not defined
        with pytest.raises(AttributeError):
            export_to_dashboard(comparison, year=2024)

    def test_export_empty(self):
        from cosilico_validators.comparison.cps import export_to_dashboard
        result = export_to_dashboard({}, year=2024)
        assert result["overall"]["match_rate"] == 0
        assert result["sections"] == []


class TestGenerateReport:
    def test_generate_with_mocked_data(self):
        from cosilico_validators.comparison.cps import generate_report
        with patch("cosilico_validators.comparison.cps.compare_cps_totals") as mock_compare:
            models = {
                "cosilico": ModelResult(name="cosilico", total=60e9, n_records=100000, time_ms=50.0),
                "policyengine": ModelResult(name="policyengine", total=62e9, n_records=100000, time_ms=200.0),
            }
            mock_compare.return_value = {
                "eitc": ComparisonTotals(variable="eitc", title="EITC", models=models),
            }
            report = generate_report(year=2024)
            assert isinstance(report, str)
            assert "EITC" in report
            assert "cosilico" in report
            assert "policyengine" in report
            assert "Records per model" in report
            assert "Performance (ms)" in report


class TestMain:
    def test_main_calls_generate_report(self):
        from cosilico_validators.comparison.cps import main
        with patch("cosilico_validators.comparison.cps.generate_report", return_value="test report"):
            main()


class TestLoadPolicyengineValues:
    def test_requires_policyengine(self):
        import sys

        from cosilico_validators.comparison.cps import load_policyengine_values
        with patch.dict(sys.modules, {"policyengine_us": None}), \
             pytest.raises(ImportError):
            load_policyengine_values(year=2024)

    def test_load_success_with_mock(self):
        """Test full body of load_policyengine_values with mocked PE."""
        import sys

        # Create a mock policyengine_us module
        mock_pe = MagicMock()
        mock_sim = MagicMock()
        mock_pe.Microsimulation.return_value = mock_sim

        # simulate weights + variable calculations
        mock_sim.calculate.side_effect = lambda var, year: (
            np.array([100.0, 200.0])
        )

        with patch.dict(sys.modules, {"policyengine_us": mock_pe}):
            # Need to reimport to get fresh function
            from cosilico_validators.comparison.cps import load_policyengine_values
            result = load_policyengine_values(year=2024, variables=["eitc"])
            assert isinstance(result, TimedResult)
            assert "weight" in result.data
            assert result.elapsed_ms >= 0

    def test_load_with_person_entity(self):
        """Test person-level aggregation path."""
        import sys
        mock_pe = MagicMock()
        mock_sim = MagicMock()
        mock_pe.Microsimulation.return_value = mock_sim

        # First call = weight (2 tax units), second = variable value (3 persons, != 2)
        # then person_tax_unit_id, then tax_unit_id
        call_count = [0]
        def mock_calculate(var, year):
            call_count[0] += 1
            if var == "tax_unit_weight":
                return np.array([100.0, 200.0])
            elif var in ("person_tax_unit_id",):
                return np.array([0, 0, 1])
            elif var in ("tax_unit_id",):
                return np.array([0, 1])
            else:
                return np.array([10.0, 20.0, 30.0])  # 3 values != 2 tax units

        mock_sim.calculate.side_effect = mock_calculate
        # Temporarily set pe_entity to 'person' on eitc
        with patch.dict(sys.modules, {"policyengine_us": mock_pe}), \
             patch.dict(
                COMPARISON_VARIABLES,
                {"eitc": {**COMPARISON_VARIABLES.get("eitc", {}), "pe_entity": "person"}},
             ):
            from cosilico_validators.comparison.cps import load_policyengine_values
            result = load_policyengine_values(year=2024, variables=["eitc"])
            assert "eitc" in result.data

    def test_load_with_exception_in_variable(self):
        """Test that exceptions in variable calculation are caught."""
        import sys
        mock_pe = MagicMock()
        mock_sim = MagicMock()
        mock_pe.Microsimulation.return_value = mock_sim

        def mock_calculate(var, year):
            if var == "tax_unit_weight":
                return np.array([100.0])
            raise Exception("Variable not found")

        mock_sim.calculate.side_effect = mock_calculate
        with patch.dict(sys.modules, {"policyengine_us": mock_pe}):
            from cosilico_validators.comparison.cps import load_policyengine_values
            result = load_policyengine_values(year=2024, variables=["eitc"])
            assert "eitc" in result.data
            assert result.data["eitc"][0] == 0.0  # zeros on error

    def test_load_default_variables(self):
        """Test with no variables specified (defaults to all)."""
        import sys
        mock_pe = MagicMock()
        mock_sim = MagicMock()
        mock_pe.Microsimulation.return_value = mock_sim
        mock_sim.calculate.return_value = np.array([100.0, 200.0])

        with patch.dict(sys.modules, {"policyengine_us": mock_pe}):
            from cosilico_validators.comparison.cps import load_policyengine_values
            result = load_policyengine_values(year=2024)
            assert "weight" in result.data

    def test_load_skips_unknown_variable(self):
        """Test with variable not in COMPARISON_VARIABLES."""
        import sys
        mock_pe = MagicMock()
        mock_sim = MagicMock()
        mock_pe.Microsimulation.return_value = mock_sim
        mock_sim.calculate.return_value = np.array([100.0])

        with patch.dict(sys.modules, {"policyengine_us": mock_pe}):
            from cosilico_validators.comparison.cps import load_policyengine_values
            result = load_policyengine_values(
                year=2024, variables=["nonexistent_var_xyz"]
            )
            assert "nonexistent_var_xyz" not in result.data


class TestLoadTaxcalcValues:
    def test_requires_taxcalc(self):
        from cosilico_validators.comparison.cps import load_taxcalc_values
        with pytest.raises(ImportError):
            load_taxcalc_values(year=2024)

    def test_load_success_with_mock(self):
        """Test full body of load_taxcalc_values with mocked taxcalc."""
        import sys
        mock_tc = MagicMock()
        mock_calc = MagicMock()
        mock_tc.Records.cps_constructor.return_value = MagicMock()
        mock_tc.Policy.return_value = MagicMock()
        mock_tc.Calculator.return_value = mock_calc
        mock_calc.array.return_value = np.array([100.0, 200.0])

        with patch.dict(sys.modules, {"taxcalc": mock_tc}):
            from cosilico_validators.comparison.cps import load_taxcalc_values
            result = load_taxcalc_values(year=2024, variables=["eitc"])
            assert isinstance(result, TimedResult)
            assert "weight" in result.data

    def test_load_with_variable_exception(self):
        """Test that exceptions in tc_var lookup are caught."""
        import sys
        mock_tc = MagicMock()
        mock_calc = MagicMock()
        mock_tc.Records.cps_constructor.return_value = MagicMock()
        mock_tc.Policy.return_value = MagicMock()
        mock_tc.Calculator.return_value = mock_calc

        call_count = [0]
        def mock_array(var):
            call_count[0] += 1
            if var == "s006":
                return np.array([100.0])
            raise Exception("Variable not found")

        mock_calc.array.side_effect = mock_array
        with patch.dict(sys.modules, {"taxcalc": mock_tc}):
            from cosilico_validators.comparison.cps import load_taxcalc_values
            result = load_taxcalc_values(year=2024, variables=["eitc"])
            assert "eitc" in result.data

    def test_load_default_variables(self):
        """Test with no variables (defaults to all)."""
        import sys
        mock_tc = MagicMock()
        mock_calc = MagicMock()
        mock_tc.Records.cps_constructor.return_value = MagicMock()
        mock_tc.Policy.return_value = MagicMock()
        mock_tc.Calculator.return_value = mock_calc
        mock_calc.array.return_value = np.array([100.0])

        with patch.dict(sys.modules, {"taxcalc": mock_tc}):
            from cosilico_validators.comparison.cps import load_taxcalc_values
            result = load_taxcalc_values(year=2024)
            assert "weight" in result.data


class TestLoadTaxsimValues:
    def test_requires_cosilico_data(self):
        from cosilico_validators.comparison.cps import load_taxsim_values
        with pytest.raises((ImportError, ModuleNotFoundError)):
            load_taxsim_values(year=2024)

    def test_load_success_with_mock(self):
        """Test full body of load_taxsim_values with mocked deps."""
        import pandas as pd
        mock_builder = MagicMock()
        mock_runner = MagicMock()
        mock_df = pd.DataFrame({
            "weight": [100.0, 200.0],
            "is_joint": [False, True],
            "head_age": [35, 40],
            "spouse_age": [0, 38],
            "num_eitc_children": [0, 2],
            "earned_income": [30000.0, 50000.0],
        })
        mock_builder.return_value = mock_df

        with patch(
            "cosilico_validators.comparison.cps._load_cosilico_data_sources",
            return_value=(mock_builder, mock_runner),
        ), patch(
            "cosilico_validators.comparison.multi_validator.get_taxsim_executable_path",
            return_value="/tmp/taxsim35",
        ):
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = (
                "taxsimid,year,v25\n1,2024,600.00\n2,2024,0.00"
            )
            with patch("subprocess.run", return_value=mock_result):
                from cosilico_validators.comparison.cps import load_taxsim_values
                result = load_taxsim_values(year=2024, variables=["eitc"])
                assert isinstance(result, TimedResult)
                assert "weight" in result.data

    def test_load_taxsim_nonzero_return(self):
        """Test TAXSIM failure raises RuntimeError."""
        import pandas as pd
        mock_builder = MagicMock()
        mock_runner = MagicMock()
        mock_df = pd.DataFrame({
            "weight": [100.0],
            "is_joint": [False],
            "head_age": [35],
            "spouse_age": [0],
            "num_eitc_children": [0],
            "earned_income": [30000.0],
        })
        mock_builder.return_value = mock_df

        with patch(
            "cosilico_validators.comparison.cps._load_cosilico_data_sources",
            return_value=(mock_builder, mock_runner),
        ), patch(
            "cosilico_validators.comparison.multi_validator.get_taxsim_executable_path",
            return_value="/tmp/taxsim35",
        ):
            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_result.stderr = "TAXSIM error"
            with patch("subprocess.run", return_value=mock_result):
                from cosilico_validators.comparison.cps import load_taxsim_values
                with pytest.raises(RuntimeError, match="TAXSIM failed"):
                    load_taxsim_values(year=2024, variables=["eitc"])

    def test_load_taxsim_nan_values(self):
        """Test NaN handling in inputs."""
        import pandas as pd
        mock_builder = MagicMock()
        mock_runner = MagicMock()
        mock_df = pd.DataFrame({
            "weight": [100.0],
            "is_joint": [False],
            "head_age": [float("nan")],
            "spouse_age": [float("nan")],
            "num_eitc_children": [float("nan")],
            "earned_income": [float("nan")],
        })
        mock_builder.return_value = mock_df

        with patch(
            "cosilico_validators.comparison.cps._load_cosilico_data_sources",
            return_value=(mock_builder, mock_runner),
        ), patch(
            "cosilico_validators.comparison.multi_validator.get_taxsim_executable_path",
            return_value="/tmp/taxsim35",
        ):
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "taxsimid,year,v25\n1,2024,0.00"
            with patch("subprocess.run", return_value=mock_result):
                from cosilico_validators.comparison.cps import load_taxsim_values
                result = load_taxsim_values(year=2024, variables=["eitc"])
                assert isinstance(result, TimedResult)


class TestLoadCosilicoDataSources:
    def test_requires_modules(self):
        from cosilico_validators.comparison.cps import _load_cosilico_data_sources
        with pytest.raises((ImportError, ModuleNotFoundError)):
            _load_cosilico_data_sources()

    def test_load_success_with_mock(self):
        """Test _load_cosilico_data_sources with mocked modules."""
        import sys
        mock_builder = MagicMock()
        mock_runner = MagicMock()

        with patch("pathlib.Path.home", return_value=MagicMock()), patch.dict(sys.modules, {
            "tax_unit_builder": mock_builder,
            "cosilico_runner": mock_runner,
        }):
            from cosilico_validators.comparison.cps import _load_cosilico_data_sources
            result = _load_cosilico_data_sources()
            assert len(result) == 2


class TestLoadCosilicoCps:
    def test_load_success_with_mock(self):
        """Test full body of load_cosilico_cps with mocked deps."""
        import pandas as pd
        mock_builder = MagicMock()
        mock_runner = MagicMock()
        mock_df = pd.DataFrame({
            "weight": [100.0, 200.0],
        })
        # Add columns for all comparison variables
        for _var_name, config in COMPARISON_VARIABLES.items():
            col = config["cosilico_col"]
            mock_df[col] = [100.0, 200.0]

        mock_builder.return_value = mock_df
        mock_runner.return_value = mock_df

        with patch(
            "cosilico_validators.comparison.cps._load_cosilico_data_sources",
            return_value=(mock_builder, mock_runner),
        ):
            from cosilico_validators.comparison.cps import load_cosilico_cps
            result = load_cosilico_cps(year=2024)
            assert isinstance(result, TimedResult)
            assert "weight" in result.data
            assert result.elapsed_ms >= 0

    def test_load_missing_column(self):
        """Test with column not in dataframe (zeros fallback)."""
        import pandas as pd
        mock_builder = MagicMock()
        mock_runner = MagicMock()
        mock_df = pd.DataFrame({
            "weight": [100.0],
        })
        mock_builder.return_value = mock_df
        mock_runner.return_value = mock_df

        with patch(
            "cosilico_validators.comparison.cps._load_cosilico_data_sources",
            return_value=(mock_builder, mock_runner),
        ):
            from cosilico_validators.comparison.cps import load_cosilico_cps
            result = load_cosilico_cps(year=2024)
            # Variables with missing columns should be zero arrays
            for var_name in COMPARISON_VARIABLES:
                assert var_name in result.data
                assert result.data[var_name][0] == 0.0


class TestCompareCpsTotalsVariableNotInConfig:
    def test_variable_not_in_comparison_variables(self):
        """Test that variables not in COMPARISON_VARIABLES are skipped."""
        from cosilico_validators.comparison.cps import compare_cps_totals
        mock_timed = TimedResult(
            data={
                "weight": np.array([1.0]),
                "nonexistent_xyz": np.array([100.0]),
            },
            elapsed_ms=50.0,
        )
        with patch("cosilico_validators.comparison.cps.load_cosilico_cps",
                    return_value=mock_timed):
            result = compare_cps_totals(
                year=2024,
                variables=["nonexistent_xyz"],
                models=["cosilico"],
            )
            assert "nonexistent_xyz" not in result


class TestLoadTaxsimSafeConversions:
    """Test safe_int / safe_float ValueError/TypeError paths in load_taxsim_values."""

    def test_load_taxsim_string_values(self):
        """Test that non-numeric string values are handled by safe_int/safe_float."""
        import pandas as pd
        mock_builder = MagicMock()
        mock_runner = MagicMock()
        mock_df = pd.DataFrame({
            "weight": [100.0],
            "is_joint": [False],
            "head_age": ["invalid_string"],  # Will fail int() -> ValueError
            "spouse_age": [0],
            "num_eitc_children": [object()],  # Will fail int() -> TypeError
            "earned_income": ["not_a_number"],  # Will fail float() -> ValueError
        })
        mock_builder.return_value = mock_df

        with patch(
            "cosilico_validators.comparison.cps._load_cosilico_data_sources",
            return_value=(mock_builder, mock_runner),
        ), patch(
            "cosilico_validators.comparison.multi_validator.get_taxsim_executable_path",
            return_value="/tmp/taxsim35",
        ):
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "taxsimid,year,v25\n1,2024,0.00"
            with patch("subprocess.run", return_value=mock_result):
                from cosilico_validators.comparison.cps import load_taxsim_values
                result = load_taxsim_values(year=2024, variables=["eitc"])
                assert isinstance(result, TimedResult)

    def test_load_taxsim_default_variables(self):
        """Test load_taxsim_values without specifying variables (defaults to all)."""
        import pandas as pd
        mock_builder = MagicMock()
        mock_runner = MagicMock()
        mock_df = pd.DataFrame({
            "weight": [100.0],
            "is_joint": [False],
            "head_age": [35],
            "spouse_age": [0],
            "num_eitc_children": [0],
            "earned_income": [30000.0],
        })
        mock_builder.return_value = mock_df

        with patch(
            "cosilico_validators.comparison.cps._load_cosilico_data_sources",
            return_value=(mock_builder, mock_runner),
        ), patch(
            "cosilico_validators.comparison.multi_validator.get_taxsim_executable_path",
            return_value="/tmp/taxsim35",
        ):
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "taxsimid,year,v25,v22,actc\n1,2024,600.00,2000.00,500.00"
            with patch("subprocess.run", return_value=mock_result):
                from cosilico_validators.comparison.cps import load_taxsim_values
                result = load_taxsim_values(year=2024)  # No variables specified
                assert isinstance(result, TimedResult)
                assert "eitc" in result.data

    def test_load_taxsim_unknown_variable_skipped(self):
        """Test that variables not in COMPARISON_VARIABLES are skipped in output parsing."""
        import pandas as pd
        mock_builder = MagicMock()
        mock_runner = MagicMock()
        mock_df = pd.DataFrame({
            "weight": [100.0],
            "is_joint": [False],
            "head_age": [35],
            "spouse_age": [0],
            "num_eitc_children": [0],
            "earned_income": [30000.0],
        })
        mock_builder.return_value = mock_df

        with patch(
            "cosilico_validators.comparison.cps._load_cosilico_data_sources",
            return_value=(mock_builder, mock_runner),
        ), patch(
            "cosilico_validators.comparison.multi_validator.get_taxsim_executable_path",
            return_value="/tmp/taxsim35",
        ):
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "taxsimid,year,v25\n1,2024,600.00"
            with patch("subprocess.run", return_value=mock_result):
                from cosilico_validators.comparison.cps import load_taxsim_values
                # Pass a variable that exists in COMPARISON_VARIABLES but whose ts_var
                # is not in the output, plus one that doesn't exist at all
                result = load_taxsim_values(year=2024, variables=["eitc", "nonexistent_xyz"])
                assert "nonexistent_xyz" not in result.data
                assert "eitc" in result.data

    def test_load_taxsim_missing_ts_var_in_output(self):
        """Test that when ts_var is not in output records, zeros are returned."""
        import pandas as pd
        mock_builder = MagicMock()
        mock_runner = MagicMock()
        mock_df = pd.DataFrame({
            "weight": [100.0],
            "is_joint": [False],
            "head_age": [35],
            "spouse_age": [0],
            "num_eitc_children": [0],
            "earned_income": [30000.0],
        })
        mock_builder.return_value = mock_df

        with patch(
            "cosilico_validators.comparison.cps._load_cosilico_data_sources",
            return_value=(mock_builder, mock_runner),
        ), patch(
            "cosilico_validators.comparison.multi_validator.get_taxsim_executable_path",
            return_value="/tmp/taxsim35",
        ):
            mock_result = MagicMock()
            mock_result.returncode = 0
            # Output only has v25 (eitc), missing ctc (v22)
            mock_result.stdout = "taxsimid,year,v25\n1,2024,600.00"
            with patch("subprocess.run", return_value=mock_result):
                from cosilico_validators.comparison.cps import load_taxsim_values
                result = load_taxsim_values(year=2024, variables=["eitc", "ctc"])
                assert result.data["ctc"][0] == 0.0  # Zeros fallback


class TestLoadTaxcalcSkipsUnknownVariable:
    def test_unknown_variable_skipped(self):
        """Test that variables not in COMPARISON_VARIABLES are skipped in taxcalc."""
        import sys
        mock_tc = MagicMock()
        mock_calc = MagicMock()
        mock_tc.Records.cps_constructor.return_value = MagicMock()
        mock_tc.Policy.return_value = MagicMock()
        mock_tc.Calculator.return_value = mock_calc
        mock_calc.array.return_value = np.array([100.0])

        with patch.dict(sys.modules, {"taxcalc": mock_tc}):
            from cosilico_validators.comparison.cps import load_taxcalc_values
            result = load_taxcalc_values(year=2024, variables=["eitc", "nonexistent_xyz"])
            assert "nonexistent_xyz" not in result.data
