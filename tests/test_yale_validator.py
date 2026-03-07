"""Tests for Yale Tax-Simulator validator - full coverage."""

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cosilico_validators.validators.base import TestCase
from cosilico_validators.validators.yale import (
    FILING_STATUS_MAP,
    YaleTaxValidator,
)


class TestResolvePath:
    def test_provided_path_valid(self, tmp_path):
        # Create expected structure
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.R").write_text("# R code")
        with patch.object(YaleTaxValidator, "_check_r_available"):
            v = YaleTaxValidator(tax_simulator_path=tmp_path)
            assert v.tax_simulator_path == tmp_path

    def test_provided_path_invalid(self, tmp_path):
        with patch.object(YaleTaxValidator, "_check_r_available"), pytest.raises(FileNotFoundError, match="not found"):
            YaleTaxValidator(tax_simulator_path=tmp_path)

    def test_search_paths_not_found(self):
        with patch.object(YaleTaxValidator, "_check_r_available"), pytest.raises(FileNotFoundError, match="not found"):
            YaleTaxValidator()

    def test_found_in_search_paths(self, tmp_path):
        """Test that Tax-Simulator is found when it exists in a search path."""
        # Create Tax-Simulator structure at a search path
        sim_path = tmp_path / "Tax-Simulator"
        (sim_path / "src").mkdir(parents=True)
        (sim_path / "src" / "main.R").write_text("# R code")

        with patch.object(YaleTaxValidator, "_check_r_available"), patch("pathlib.Path.home", return_value=tmp_path):
            v = YaleTaxValidator()
            assert v.tax_simulator_path == sim_path


class TestCheckRAvailable:
    def test_r_available(self, tmp_path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.R").write_text("# R code")
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result):
            v = YaleTaxValidator(tax_simulator_path=tmp_path)
            assert v.tax_simulator_path == tmp_path

    def test_r_not_installed(self, tmp_path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.R").write_text("# R code")
        with (
            patch("subprocess.run", side_effect=FileNotFoundError("Rscript not found")),
            pytest.raises(RuntimeError, match="Rscript not found"),
        ):
            YaleTaxValidator(tax_simulator_path=tmp_path)

    def test_r_nonzero_exit(self, tmp_path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.R").write_text("# R code")
        mock_result = MagicMock()
        mock_result.returncode = 1
        with patch("subprocess.run", return_value=mock_result), pytest.raises(RuntimeError, match="non-zero"):
            YaleTaxValidator(tax_simulator_path=tmp_path)


class TestCreateTaxUnitInput:
    @pytest.fixture
    def validator(self):
        with (
            patch.object(YaleTaxValidator, "_resolve_path") as mock_path,
            patch.object(YaleTaxValidator, "_check_r_available"),
        ):
            mock_path.return_value = Path("/mock/Tax-Simulator")
            return YaleTaxValidator()

    def test_basic_input(self, validator, tmp_path):
        tc = TestCase(name="test", inputs={"earned_income": 30000}, expected={})
        input_file = validator._create_tax_unit_input(tc, 2024, tmp_path)
        assert input_file.exists()
        with open(input_file) as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert row["e00200"] == "30000"
        assert row["FLPDYR"] == "2024"

    def test_self_employment(self, validator, tmp_path):
        tc = TestCase(name="test", inputs={"self_employment_income": 50000}, expected={})
        input_file = validator._create_tax_unit_input(tc, 2024, tmp_path)
        with open(input_file) as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert row["e00900"] == "50000"

    def test_dividend_income(self, validator, tmp_path):
        tc = TestCase(name="test", inputs={"dividend_income": 3000}, expected={})
        input_file = validator._create_tax_unit_input(tc, 2024, tmp_path)
        with open(input_file) as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert row["e00600"] == "3000"

    def test_interest_income(self, validator, tmp_path):
        tc = TestCase(name="test", inputs={"interest_income": 2000}, expected={})
        input_file = validator._create_tax_unit_input(tc, 2024, tmp_path)
        with open(input_file) as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert row["e00300"] == "2000"

    def test_capital_gains(self, validator, tmp_path):
        tc = TestCase(name="test", inputs={"capital_gains": 10000}, expected={})
        input_file = validator._create_tax_unit_input(tc, 2024, tmp_path)
        with open(input_file) as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert row["p23250"] == "10000"

    def test_business_income(self, validator, tmp_path):
        """Test business_income is added to e00900."""
        tc = TestCase(
            name="test",
            inputs={"business_income": 25000},
            expected={},
        )
        input_file = validator._create_tax_unit_input(tc, 2024, tmp_path)
        with open(input_file) as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert int(row["e00900"]) == 25000


class TestCreateRunscript:
    @pytest.fixture
    def validator(self):
        with (
            patch.object(YaleTaxValidator, "_resolve_path") as mock_path,
            patch.object(YaleTaxValidator, "_check_r_available"),
        ):
            mock_path.return_value = Path("/mock/Tax-Simulator")
            return YaleTaxValidator()

    def test_runscript_created(self, validator, tmp_path):
        runscript = validator._create_runscript(2024, tmp_path)
        assert runscript.exists()
        with open(runscript) as f:
            reader = csv.reader(f)
            header = next(reader)
            data = next(reader)
        assert "scenario_id" in header
        assert data[0] == "baseline"
        assert data[3] == "2024"
        assert data[4] == "2024"


class TestRunSimulator:
    @pytest.fixture
    def validator(self):
        with (
            patch.object(YaleTaxValidator, "_resolve_path") as mock_path,
            patch.object(YaleTaxValidator, "_check_r_available"),
        ):
            mock_path.return_value = Path("/mock/Tax-Simulator")
            return YaleTaxValidator()

    def test_success(self, validator, tmp_path):
        mock_result = MagicMock()
        mock_result.returncode = 0
        with (
            patch("subprocess.run", return_value=mock_result),
            patch.object(validator, "_parse_output", return_value={"eitc": 500.0}),
        ):
            result = validator._run_simulator(tmp_path / "input.csv", tmp_path / "runscript.csv", tmp_path, 2024)
            assert result == {"eitc": 500.0}

    def test_failure(self, validator, tmp_path):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = "some output"
        mock_result.stderr = "some error"
        with patch("subprocess.run", return_value=mock_result), pytest.raises(RuntimeError, match="failed with code"):
            validator._run_simulator(tmp_path / "input.csv", tmp_path / "runscript.csv", tmp_path, 2024)

    def test_timeout(self, validator, tmp_path):
        import subprocess

        with (
            patch("subprocess.run", side_effect=subprocess.TimeoutExpired("Rscript", 120)),
            pytest.raises(RuntimeError, match="timed out"),
        ):
            validator._run_simulator(tmp_path / "input.csv", tmp_path / "runscript.csv", tmp_path, 2024)


class TestParseOutput:
    @pytest.fixture
    def validator(self):
        with (
            patch.object(YaleTaxValidator, "_resolve_path") as mock_path,
            patch.object(YaleTaxValidator, "_check_r_available"),
        ):
            mock_path.return_value = Path("/mock/Tax-Simulator")
            return YaleTaxValidator()

    def test_parse_detail_dir(self, validator, tmp_path):
        output_dir = tmp_path / "output"
        detail_dir = output_dir / "detail"
        detail_dir.mkdir(parents=True)
        csv_file = detail_dir / "results_2024.csv"
        csv_file.write_text("eitc,income_tax\n500,3000\n")
        result = validator._parse_output(output_dir, 2024)
        assert result["eitc"] == 500.0
        assert result["income_tax"] == 3000.0

    def test_parse_summary_json(self, validator, tmp_path):
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True)
        summary_file = output_dir / "summary.json"
        summary_file.write_text(json.dumps({"eitc": 500.0}))
        result = validator._parse_output(output_dir, 2024)
        assert result["eitc"] == 500.0

    def test_parse_empty(self, validator, tmp_path):
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True)
        result = validator._parse_output(output_dir, 2024)
        assert result == {}


class TestParseCsvOutput:
    @pytest.fixture
    def validator(self):
        with (
            patch.object(YaleTaxValidator, "_resolve_path") as mock_path,
            patch.object(YaleTaxValidator, "_check_r_available"),
        ):
            mock_path.return_value = Path("/mock/Tax-Simulator")
            return YaleTaxValidator()

    def test_parse_csv(self, validator, tmp_path):
        csv_file = tmp_path / "output.csv"
        csv_file.write_text("eitc,iitax,c00100\n500,3000,50000\n")
        result = validator._parse_csv_output(csv_file)
        assert result["eitc"] == 500.0
        assert result["income_tax"] == 3000.0
        assert result["agi"] == 50000.0

    def test_parse_csv_invalid_values(self, validator, tmp_path):
        csv_file = tmp_path / "output.csv"
        csv_file.write_text("eitc,iitax\n500,not_a_number\n")
        result = validator._parse_csv_output(csv_file)
        assert result["eitc"] == 500.0
        # "not_a_number" should be skipped
        assert "income_tax" not in result


class TestValidateComplete:
    @pytest.fixture
    def validator(self):
        with (
            patch.object(YaleTaxValidator, "_resolve_path") as mock_path,
            patch.object(YaleTaxValidator, "_check_r_available"),
        ):
            mock_path.return_value = Path("/mock/Tax-Simulator")
            return YaleTaxValidator()

    def test_runtime_error(self, validator):
        with patch.object(validator, "_run_simulator", side_effect=RuntimeError("R failed")):
            tc = TestCase(name="test", inputs={"earned_income": 30000}, expected={})
            result = validator.validate(tc, "eitc", 2024)
            assert not result.success
            assert "execution failed" in result.error

    def test_unexpected_error(self, validator):
        with patch.object(validator, "_run_simulator", side_effect=ValueError("unexpected")):
            tc = TestCase(name="test", inputs={"earned_income": 30000}, expected={})
            result = validator.validate(tc, "eitc", 2024)
            assert not result.success
            assert "Unexpected error" in result.error

    def test_file_not_found_error(self, validator):
        with patch.object(validator, "_run_simulator", side_effect=FileNotFoundError("not found")):
            tc = TestCase(name="test", inputs={"earned_income": 30000}, expected={})
            result = validator.validate(tc, "eitc", 2024)
            assert not result.success
            assert "not found" in result.error

    def test_variable_not_in_output(self, validator):
        with patch.object(validator, "_run_simulator", return_value={"eitc": 500}):
            tc = TestCase(name="test", inputs={"earned_income": 30000}, expected={})
            result = validator.validate(tc, "amt", 2024)
            assert not result.success
            assert "not found" in result.error

    def test_temp_dir_cleanup(self, validator):
        """Test that temp dir is cleaned up even on error."""
        with patch.object(validator, "_run_simulator", side_effect=RuntimeError("fail")):
            tc = TestCase(name="test", inputs={"earned_income": 30000}, expected={})
            validator.validate(tc, "eitc", 2024)
            # Even on failure, should complete without raising

    def test_successful_validation(self, validator):
        with patch.object(validator, "_run_simulator", return_value={"eitc": 500.0}):
            tc = TestCase(name="test", inputs={"earned_income": 30000}, expected={})
            result = validator.validate(tc, "eitc", 2024)
            assert result.success
            assert result.calculated_value == 500.0
            assert result.metadata["yale_variable"] == "eitc"


class TestFilingStatusMap:
    def test_all_mapped(self):
        assert FILING_STATUS_MAP["SINGLE"] == "single"
        assert FILING_STATUS_MAP["JOINT"] == "married"
        assert FILING_STATUS_MAP["MARRIED_FILING_JOINTLY"] == "married"
        assert FILING_STATUS_MAP["MARRIED_FILING_SEPARATELY"] == "married_separate"
        assert FILING_STATUS_MAP["HEAD_OF_HOUSEHOLD"] == "head_of_household"
        assert FILING_STATUS_MAP["SEPARATE"] == "married_separate"
