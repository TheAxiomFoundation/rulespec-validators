"""Tests for comparison/multi_validator.py module."""

from unittest.mock import MagicMock, patch

import numpy as np

from cosilico_validators.comparison.multi_validator import (
    TAXSIM_DOWNLOAD_URLS,
    MultiValidatorResult,
    ValidatorComparison,
    compare_microdata,
    compare_single_case,
    get_taxsim_executable_path,
)
from cosilico_validators.validators.base import TestCase


class TestTaxsimDownloadUrls:
    def test_urls_exist(self):
        assert "darwin" in TAXSIM_DOWNLOAD_URLS
        assert "linux" in TAXSIM_DOWNLOAD_URLS
        assert "windows" in TAXSIM_DOWNLOAD_URLS


class TestValidatorComparison:
    def test_creation(self):
        vc = ValidatorComparison(
            variable="eitc",
            cosilico_value=500.0,
            validator_results={"PE": 510.0, "TAXSIM": 505.0},
            differences={"PE": -10.0, "TAXSIM": -5.0},
            match_flags={"PE": True, "TAXSIM": True},
        )
        assert vc.variable == "eitc"
        assert vc.cosilico_value == 500.0
        assert vc.validator_results["PE"] == 510.0


class TestMultiValidatorResult:
    def test_creation(self):
        mvr = MultiValidatorResult(
            variable="eitc",
            n_records=1000,
            validators_used=["PE", "TAXSIM"],
            match_rates={"PE": 0.95, "TAXSIM": 0.90},
            mean_errors={"PE": 5.0, "TAXSIM": 10.0},
            weighted_totals={"cosilico": 60e9, "PE": 62e9},
        )
        assert mvr.variable == "eitc"
        assert mvr.n_records == 1000


class TestCompareSingleCase:
    def test_compare_with_mocked_validators(self):
        test_case = TestCase(
            name="test",
            inputs={"earned_income": 20000, "filing_status": "SINGLE"},
            expected={"eitc": 500},
        )

        with (
            patch("cosilico_validators.comparison.multi_validator.PolicyEngineValidator") as mock_pe,
            patch("cosilico_validators.comparison.multi_validator.TaxsimValidator") as mock_ts,
            patch("cosilico_validators.comparison.multi_validator.TaxCalculatorValidator") as mock_tc,
        ):
            # Mock PE
            mock_pe_inst = MagicMock()
            mock_pe.return_value = mock_pe_inst
            mock_pe_result = MagicMock()
            mock_pe_result.success = True
            mock_pe_result.calculated_value = 510.0
            mock_pe_inst.validate.return_value = mock_pe_result

            # Mock TAXSIM
            mock_ts_inst = MagicMock()
            mock_ts.return_value = mock_ts_inst
            mock_ts_result = MagicMock()
            mock_ts_result.success = True
            mock_ts_result.calculated_value = 505.0
            mock_ts_inst.validate.return_value = mock_ts_result

            # Mock TaxCalc
            mock_tc_inst = MagicMock()
            mock_tc.return_value = mock_tc_inst
            mock_tc_result = MagicMock()
            mock_tc_result.success = True
            mock_tc_result.calculated_value = 500.0
            mock_tc_inst.validate.return_value = mock_tc_result

            with patch(
                "cosilico_validators.comparison.multi_validator.get_taxsim_executable_path", return_value="/tmp/taxsim"
            ):
                result = compare_single_case(
                    test_case=test_case,
                    cosilico_value=500.0,
                    variable="eitc",
                    year=2023,
                    taxsim_mode="local",
                )
                assert isinstance(result, ValidatorComparison)
                assert result.variable == "eitc"

    def test_compare_with_failed_validator(self):
        test_case = TestCase(
            name="test",
            inputs={"earned_income": 20000},
            expected={"eitc": 500},
        )

        with patch("cosilico_validators.comparison.multi_validator.PolicyEngineValidator") as mock_pe:
            mock_pe.side_effect = Exception("PE not installed")

            with patch("cosilico_validators.comparison.multi_validator.TaxsimValidator") as mock_ts:
                mock_ts.side_effect = Exception("TAXSIM not available")

                with patch("cosilico_validators.comparison.multi_validator.TaxCalculatorValidator") as mock_tc:
                    mock_tc.side_effect = Exception("TaxCalc not available")

                    result = compare_single_case(
                        test_case=test_case,
                        cosilico_value=500.0,
                        variable="eitc",
                        year=2023,
                    )
                    assert isinstance(result, ValidatorComparison)
                    # All should have failed
                    for val in result.validator_results.values():
                        assert val is None

    def test_compare_web_taxsim(self):
        test_case = TestCase(
            name="test",
            inputs={"earned_income": 20000},
            expected={"eitc": 500},
        )

        with (
            patch("cosilico_validators.comparison.multi_validator.PolicyEngineValidator") as mock_pe,
            patch("cosilico_validators.comparison.multi_validator.TaxsimValidator") as mock_ts,
            patch("cosilico_validators.comparison.multi_validator.TaxCalculatorValidator") as mock_tc,
        ):
            for mock in [mock_pe, mock_ts, mock_tc]:
                inst = MagicMock()
                mock.return_value = inst
                mock_result = MagicMock()
                mock_result.success = False
                mock_result.calculated_value = None
                inst.validate.return_value = mock_result

            result = compare_single_case(
                test_case=test_case,
                cosilico_value=500.0,
                variable="eitc",
                year=2023,
                taxsim_mode="web",
            )
            assert isinstance(result, ValidatorComparison)


class TestCompareMicrodata:
    def test_compare_microdata_basic(self):
        cosilico_values = np.array([500.0, 600.0, 0.0])

        def input_builder(i):
            return TestCase(
                name=f"case_{i}",
                inputs={"earned_income": 20000 + i * 10000},
                expected={},
            )

        # Mock all validators to succeed
        with (
            patch("cosilico_validators.comparison.multi_validator.TaxsimValidator") as mock_ts,
            patch("cosilico_validators.comparison.multi_validator.TaxCalculatorValidator") as mock_tc,
        ):
            # Setup TAXSIM mock
            mock_ts_inst = MagicMock()
            mock_ts.return_value = mock_ts_inst
            mock_ts_result = MagicMock()
            mock_ts_result.success = True
            mock_ts_result.calculated_value = 510.0
            mock_ts_inst.batch_validate.return_value = [mock_ts_result] * 3

            # Setup TaxCalc mock
            mock_tc_inst = MagicMock()
            mock_tc.return_value = mock_tc_inst
            mock_tc_result = MagicMock()
            mock_tc_result.success = True
            mock_tc_result.calculated_value = 505.0
            mock_tc_inst.batch_validate.return_value = [mock_tc_result] * 3

            with patch(
                "cosilico_validators.comparison.multi_validator.get_taxsim_executable_path", return_value="/tmp/taxsim"
            ):
                result = compare_microdata(
                    cosilico_values=cosilico_values,
                    input_builder=input_builder,
                    variable="eitc",
                    year=2023,
                    taxsim_mode="local",
                )
                assert isinstance(result, MultiValidatorResult)
                assert result.variable == "eitc"
                assert result.n_records == 3
                assert "taxsim" in result.match_rates
                assert "taxcalc" in result.match_rates

    def test_compare_microdata_with_sample_size(self):
        cosilico_values = np.array([500.0, 600.0, 700.0, 800.0])

        def input_builder(i):
            return TestCase(name=f"case_{i}", inputs={}, expected={})

        with patch("cosilico_validators.comparison.multi_validator.TaxsimValidator") as mock_ts:
            mock_ts_inst = MagicMock()
            mock_ts.return_value = mock_ts_inst
            mock_ts_result = MagicMock()
            mock_ts_result.success = True
            mock_ts_result.calculated_value = 505.0
            mock_ts_inst.batch_validate.return_value = [mock_ts_result] * 2

            with patch(
                "cosilico_validators.comparison.multi_validator.get_taxsim_executable_path", return_value="/tmp/taxsim"
            ):
                result = compare_microdata(
                    cosilico_values=cosilico_values,
                    input_builder=input_builder,
                    variable="eitc",
                    year=2023,
                    sample_size=2,
                    validators=["taxsim"],
                    taxsim_mode="local",
                )
                assert result.n_records == 2

    def test_compare_microdata_no_valid_results(self):
        cosilico_values = np.array([500.0])

        def input_builder(i):
            return TestCase(name=f"case_{i}", inputs={}, expected={})

        with patch("cosilico_validators.comparison.multi_validator.TaxsimValidator") as mock_ts:
            mock_ts_inst = MagicMock()
            mock_ts.return_value = mock_ts_inst
            mock_ts_result = MagicMock()
            mock_ts_result.success = False
            mock_ts_result.calculated_value = None
            mock_ts_inst.batch_validate.return_value = [mock_ts_result]

            with patch(
                "cosilico_validators.comparison.multi_validator.get_taxsim_executable_path", return_value="/tmp/taxsim"
            ):
                result = compare_microdata(
                    cosilico_values=cosilico_values,
                    input_builder=input_builder,
                    variable="eitc",
                    year=2023,
                    validators=["taxsim"],
                    taxsim_mode="local",
                )
                assert result.match_rates["taxsim"] == 0.0

    def test_compare_microdata_web_mode(self):
        cosilico_values = np.array([500.0])

        def input_builder(i):
            return TestCase(name=f"case_{i}", inputs={}, expected={})

        with patch("cosilico_validators.comparison.multi_validator.TaxsimValidator") as mock_ts:
            mock_ts_inst = MagicMock()
            mock_ts.return_value = mock_ts_inst
            mock_ts_result = MagicMock()
            mock_ts_result.success = True
            mock_ts_result.calculated_value = 500.0
            mock_ts_inst.batch_validate.return_value = [mock_ts_result]

            result = compare_microdata(
                cosilico_values=cosilico_values,
                input_builder=input_builder,
                variable="eitc",
                year=2023,
                validators=["taxsim"],
                taxsim_mode="web",
            )
            assert result.match_rates["taxsim"] == 1.0

    def test_compare_microdata_with_pe(self):
        cosilico_values = np.array([500.0])

        def input_builder(i):
            return TestCase(name=f"case_{i}", inputs={}, expected={})

        with patch("cosilico_validators.comparison.multi_validator.PolicyEngineValidator") as mock_pe:
            mock_pe_inst = MagicMock()
            mock_pe.return_value = mock_pe_inst
            mock_pe_result = MagicMock()
            mock_pe_result.success = True
            mock_pe_result.calculated_value = 510.0
            mock_pe_inst.batch_validate.return_value = [mock_pe_result]

            result = compare_microdata(
                cosilico_values=cosilico_values,
                input_builder=input_builder,
                variable="eitc",
                year=2023,
                validators=["policyengine"],
            )
            assert "policyengine" in result.match_rates

    def test_compare_microdata_init_failure(self):
        cosilico_values = np.array([500.0])

        def input_builder(i):
            return TestCase(name=f"case_{i}", inputs={}, expected={})

        with (
            patch(
                "cosilico_validators.comparison.multi_validator.TaxsimValidator", side_effect=Exception("init failed")
            ),
            patch(
                "cosilico_validators.comparison.multi_validator.get_taxsim_executable_path", return_value="/tmp/taxsim"
            ),
        ):
            result = compare_microdata(
                cosilico_values=cosilico_values,
                input_builder=input_builder,
                variable="eitc",
                year=2023,
                validators=["taxsim"],
                taxsim_mode="local",
            )
            # No validators succeeded
            assert result.validators_used == []

    def test_compare_microdata_without_batch(self):
        """Test fallback to single validation when batch_validate not available."""
        cosilico_values = np.array([500.0])

        def input_builder(i):
            return TestCase(name=f"case_{i}", inputs={}, expected={})

        with patch("cosilico_validators.comparison.multi_validator.TaxCalculatorValidator") as mock_tc:
            mock_tc_inst = MagicMock(spec=[])  # No batch_validate
            mock_tc_inst.validate = MagicMock()
            mock_tc.return_value = mock_tc_inst
            mock_tc_result = MagicMock()
            mock_tc_result.success = True
            mock_tc_result.calculated_value = 500.0
            mock_tc_inst.validate.return_value = mock_tc_result
            # Remove batch_validate
            del mock_tc_inst.batch_validate

            result = compare_microdata(
                cosilico_values=cosilico_values,
                input_builder=input_builder,
                variable="eitc",
                year=2023,
                validators=["taxcalc"],
            )
            assert "taxcalc" in result.match_rates


class TestRunComparisonDemo:
    def test_demo_runs(self):
        with patch("cosilico_validators.comparison.multi_validator.compare_single_case") as mock_compare:
            mock_compare.return_value = ValidatorComparison(
                variable="eitc",
                cosilico_value=560.0,
                validator_results={"policyengine": 570.0, "taxsim": None},
                differences={"policyengine": -10.0, "taxsim": None},
                match_flags={"policyengine": True, "taxsim": False},
            )
            from cosilico_validators.comparison.multi_validator import run_comparison_demo

            run_comparison_demo(year=2023)


class TestGetTaxsimExecutablePath:
    def test_path_creation(self, tmp_path):
        with (
            patch("cosilico_validators.comparison.multi_validator.Path.home", return_value=tmp_path),
            patch("cosilico_validators.comparison.multi_validator.urllib.request.urlretrieve"),
            patch("cosilico_validators.comparison.multi_validator.os.chmod"),
        ):
            path = get_taxsim_executable_path()
            assert isinstance(path, type(tmp_path / "test"))

    def test_existing_executable(self, tmp_path):
        cache_dir = tmp_path / ".cache" / "cosilico-validators" / "taxsim"
        cache_dir.mkdir(parents=True)
        (cache_dir / "taxsimtest-osx.exe").write_text("fake")
        with patch("cosilico_validators.comparison.multi_validator.Path.home", return_value=tmp_path):
            path = get_taxsim_executable_path()
            assert path.exists()

    def test_linux_platform(self, tmp_path):
        with (
            patch("cosilico_validators.comparison.multi_validator.Path.home", return_value=tmp_path),
            patch("cosilico_validators.comparison.multi_validator.platform.system", return_value="Linux"),
            patch("cosilico_validators.comparison.multi_validator.urllib.request.urlretrieve"),
            patch("cosilico_validators.comparison.multi_validator.os.chmod"),
        ):
            path = get_taxsim_executable_path()
            assert "linux" in str(path)

    def test_windows_platform(self, tmp_path):
        with (
            patch("cosilico_validators.comparison.multi_validator.Path.home", return_value=tmp_path),
            patch("cosilico_validators.comparison.multi_validator.platform.system", return_value="Windows"),
            patch("cosilico_validators.comparison.multi_validator.urllib.request.urlretrieve"),
            patch("cosilico_validators.comparison.multi_validator.os.chmod"),
        ):
            path = get_taxsim_executable_path()
            assert "windows" in str(path)


class TestCompareSingleCaseUnknownValidator:
    def test_unknown_validator_skipped(self):
        """Test that unknown validator names are skipped with continue."""
        test_case = TestCase(
            name="test",
            inputs={"earned_income": 20000},
            expected={"eitc": 500},
        )
        result = compare_single_case(
            test_case=test_case,
            cosilico_value=500.0,
            variable="eitc",
            year=2023,
            validators=["unknown_validator"],
        )
        assert isinstance(result, ValidatorComparison)
        # No validators should have been called
        assert len(result.validator_results) == 0
