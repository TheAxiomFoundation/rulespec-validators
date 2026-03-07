"""Tests for TAXSIM validator - full coverage."""

import os
from unittest.mock import MagicMock, patch

import pytest

from cosilico_validators.validators.base import TestCase
from cosilico_validators.validators.taxsim import (
    TaxsimValidator,
)


class TestResolvePathLocal:
    def test_provided_path_exists(self, tmp_path):
        exe = tmp_path / "taxsim35"
        exe.write_text("fake")
        v = TaxsimValidator.__new__(TaxsimValidator)
        result = v._resolve_taxsim_path(str(exe))
        assert result == exe

    def test_provided_path_not_found(self):
        v = TaxsimValidator.__new__(TaxsimValidator)
        with pytest.raises(FileNotFoundError):
            v._resolve_taxsim_path("/nonexistent/taxsim35")

    def test_search_paths_finds_existing(self, tmp_path):
        """When the taxsim35 exe exists in a search path, it should be found."""
        import platform

        system = platform.system().lower()
        if system == "darwin":
            exe_name = "taxsim35-osx.exe"
        elif system == "windows":
            exe_name = "taxsim35-windows.exe"
        else:
            exe_name = "taxsim35-unix.exe"

        # Create a fake executable in the home-based search path
        taxsim_dir = tmp_path / ".cosilico" / "taxsim"
        taxsim_dir.mkdir(parents=True)
        exe_file = taxsim_dir / exe_name
        exe_file.write_text("fake")

        v = TaxsimValidator.__new__(TaxsimValidator)
        with patch("pathlib.Path.home", return_value=tmp_path):
            result = v._resolve_taxsim_path(None)
        assert result.exists()

    def test_local_mode_init(self, tmp_path):
        exe = tmp_path / "taxsim35-osx.exe"
        exe.write_text("fake")
        v = TaxsimValidator(mode="local", taxsim_path=str(exe))
        assert v.mode == "local"
        assert v.taxsim_path == exe


class TestCreateInputCsv:
    def test_create_csv_file(self):
        v = TaxsimValidator()
        taxsim_input = {
            "taxsimid": 1,
            "year": 2023,
            "mstat": 1,
            "pwages": 30000,
            "idtl": 2,
        }
        csv_file = v._create_input_csv(taxsim_input)
        assert os.path.exists(csv_file)
        with open(csv_file) as f:
            content = f.read()
        assert "taxsimid" in content
        assert "30000" in content
        os.unlink(csv_file)


class TestExecuteWeb:
    def test_success(self):
        v = TaxsimValidator(max_retries=1, timeout=10)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "taxsimid,year,fiitax,v25\n1,2023,5000.00,600.00"
        with patch("subprocess.run", return_value=mock_result):
            output = v._execute_web("csv data")
            assert "fiitax" in output

    def test_curl_error(self):
        v = TaxsimValidator(max_retries=1, timeout=10)
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "connection refused"
        with patch("subprocess.run", return_value=mock_result), pytest.raises(RuntimeError, match="curl error"):
            v._execute_web("csv data")

    def test_empty_response(self):
        v = TaxsimValidator(max_retries=1, timeout=10)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        with patch("subprocess.run", return_value=mock_result), pytest.raises(RuntimeError, match="Empty response"):
            v._execute_web("csv data")

    def test_html_error_response(self):
        v = TaxsimValidator(max_retries=1, timeout=10)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "<html><body>Error</body></html>"
        with patch("subprocess.run", return_value=mock_result), pytest.raises(RuntimeError, match="API error"):
            v._execute_web("csv data")

    def test_timeout_with_retries(self):
        import subprocess

        v = TaxsimValidator(max_retries=2, timeout=10)
        with (
            patch("subprocess.run", side_effect=subprocess.TimeoutExpired("curl", 10)),
            patch("time.sleep"),
            pytest.raises(RuntimeError, match="timeout"),
        ):
            v._execute_web("csv data")

    def test_retry_on_error(self):
        v = TaxsimValidator(max_retries=2, timeout=10)
        mock_success = MagicMock()
        mock_success.returncode = 0
        mock_success.stdout = "taxsimid,year,fiitax\n1,2023,5000"
        with patch("subprocess.run", side_effect=[Exception("first fail"), mock_success]), patch("time.sleep"):
            output = v._execute_web("csv data")
            assert "fiitax" in output


class TestExecuteLocal:
    def test_no_path(self):
        v = TaxsimValidator(mode="web")
        with pytest.raises(RuntimeError, match="requires TAXSIM executable path"):
            v._execute_local("input.csv")

    def test_success(self, tmp_path):
        exe = tmp_path / "taxsim35"
        exe.write_text("fake")
        v = TaxsimValidator.__new__(TaxsimValidator)
        v.taxsim_path = exe
        v.mode = "local"
        mock_result = MagicMock()
        mock_result.returncode = 0
        with (
            patch("subprocess.run", return_value=mock_result),
            patch("builtins.open", MagicMock()),
            patch("os.path.exists", return_value=True),
            patch("os.unlink"),
            patch("os.close"),
            patch("os.chmod"),
        ):
            # Need to patch the tempfile creation and file reading
            import tempfile

            with patch.object(tempfile, "mkstemp", return_value=(5, "/tmp/output.csv")):
                # Patch the open for reading output
                mock_open = MagicMock()
                mock_file = MagicMock()
                mock_file.read.return_value = "taxsimid,year,fiitax\n1,2023,5000"
                mock_open.return_value.__enter__ = MagicMock(return_value=mock_file)
                mock_open.return_value.__exit__ = MagicMock(return_value=False)
                with patch("builtins.open", mock_open):
                    output = v._execute_local("/tmp/input.csv")
                    assert "fiitax" in output

    def test_failure(self, tmp_path):
        exe = tmp_path / "taxsim35"
        exe.write_text("fake")
        v = TaxsimValidator.__new__(TaxsimValidator)
        v.taxsim_path = exe
        v.mode = "local"
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "error occurred"
        import tempfile

        with (
            patch("subprocess.run", return_value=mock_result),
            patch("os.close"),
            patch("os.chmod"),
            patch("os.path.exists", return_value=True),
            patch("os.unlink"),
            patch.object(tempfile, "mkstemp", return_value=(5, "/tmp/output.csv")),
            pytest.raises(RuntimeError, match="TAXSIM failed"),
        ):
            v._execute_local("/tmp/input.csv")


class TestValidateComplete:
    def test_web_mode_success(self):
        v = TaxsimValidator(mode="web", max_retries=1)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "taxsimid,year,fiitax,v25\n1,2023,5000.00,600.00"
        with patch("subprocess.run", return_value=mock_result):
            tc = TestCase(name="test", inputs={"earned_income": 30000}, expected={})
            result = v.validate(tc, "eitc", 2023)
            assert result.success
            assert result.calculated_value == 600.0
            assert result.metadata["mode"] == "web"

    def test_web_mode_variable_not_found(self):
        v = TaxsimValidator(mode="web", max_retries=1)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "taxsimid,year,fiitax\n1,2023,5000.00"
        with patch("subprocess.run", return_value=mock_result):
            tc = TestCase(name="test", inputs={"earned_income": 30000}, expected={})
            result = v.validate(tc, "eitc", 2023)
            assert not result.success
            assert "Could not find" in result.error

    def test_file_not_found_error(self):
        v = TaxsimValidator(mode="web", max_retries=1)
        with patch.object(v, "_execute_web", side_effect=FileNotFoundError("curl not found")):
            tc = TestCase(name="test", inputs={"earned_income": 30000}, expected={})
            result = v.validate(tc, "eitc", 2023)
            assert not result.success
            assert "curl not found" in result.error

    def test_generic_exception(self):
        v = TaxsimValidator(mode="web", max_retries=1)
        with patch.object(v, "_execute_web", side_effect=RuntimeError("API down")):
            tc = TestCase(name="test", inputs={"earned_income": 30000}, expected={})
            result = v.validate(tc, "eitc", 2023)
            assert not result.success
            assert "TAXSIM execution failed" in result.error

    def test_local_mode_success(self, tmp_path):
        exe = tmp_path / "taxsim35"
        exe.write_text("fake")
        v = TaxsimValidator(mode="local", taxsim_path=str(exe))
        with patch.object(v, "_execute_local", return_value="taxsimid,year,fiitax,v25\n1,2023,5000.00,600.00"):
            tc = TestCase(name="test", inputs={"earned_income": 30000}, expected={})
            result = v.validate(tc, "eitc", 2023)
            assert result.success
            assert result.calculated_value == 600.0


class TestBatchValidate:
    def test_empty_batch(self):
        v = TaxsimValidator()
        results = v.batch_validate([], "eitc", 2023)
        assert results == []

    def test_invalid_year(self):
        v = TaxsimValidator()
        cases = [TestCase(name="test", inputs={}, expected={})]
        results = v.batch_validate(cases, "eitc", 1959)
        assert len(results) == 1
        assert "1960-2023" in results[0].error

    def test_unsupported_variable(self):
        v = TaxsimValidator()
        cases = [TestCase(name="test", inputs={}, expected={})]
        results = v.batch_validate(cases, "snap_xyz", 2023)
        assert len(results) == 1
        assert "not supported" in results[0].error

    def test_web_batch_success(self):
        v = TaxsimValidator(mode="web", max_retries=1)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "taxsimid,year,fiitax,v25\n1,2023,5000.00,600.00\n2,2023,3000.00,400.00"
        with patch("subprocess.run", return_value=mock_result):
            cases = [
                TestCase(name="case1", inputs={"earned_income": 30000}, expected={}),
                TestCase(name="case2", inputs={"earned_income": 20000}, expected={}),
            ]
            results = v.batch_validate(cases, "eitc", 2023)
            assert len(results) == 2
            assert results[0].calculated_value == 600.0
            assert results[1].calculated_value == 400.0

    def test_web_batch_missing_case(self):
        v = TaxsimValidator(mode="web", max_retries=1)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "taxsimid,year,fiitax,v25\n1,2023,5000.00,600.00"
        with patch("subprocess.run", return_value=mock_result):
            cases = [
                TestCase(name="case1", inputs={"earned_income": 30000}, expected={}),
                TestCase(name="case2", inputs={"earned_income": 20000}, expected={}),
            ]
            results = v.batch_validate(cases, "eitc", 2023)
            assert len(results) == 2
            assert results[0].success
            assert not results[1].success

    def test_web_batch_failure(self):
        v = TaxsimValidator(mode="web", max_retries=1)
        with patch.object(v, "_execute_web", side_effect=RuntimeError("API down")):
            cases = [
                TestCase(name="case1", inputs={"earned_income": 30000}, expected={}),
            ]
            results = v.batch_validate(cases, "eitc", 2023)
            assert len(results) == 1
            assert "batch execution failed" in results[0].error


class TestParseOutputEdgeCases:
    def test_parse_direct_lookup(self):
        v = TaxsimValidator()
        output = "taxsimid,year,eitc\n1,2023,600.00"
        result = v._parse_output(output, "eitc")
        assert result == 600.0

    def test_parse_partial_match(self):
        v = TaxsimValidator()
        output = "taxsimid,year,v25_eitc\n1,2023,600.00"
        result = v._parse_output(output, "eitc")
        assert result == 600.0

    def test_parse_invalid_output(self):
        v = TaxsimValidator()
        with pytest.raises(ValueError, match="Invalid TAXSIM output"):
            v._parse_output("only one line", "eitc")


class TestBatchValidateEdgeCases:
    def test_local_mode_batch(self, tmp_path):
        """Local mode batch delegates to base class batch_validate."""
        exe = tmp_path / "taxsim35"
        exe.write_text("fake")
        v = TaxsimValidator(mode="local", taxsim_path=str(exe))

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "taxsimid,year,fiitax,v25\n1,2023,5000.00,600.00"
        with patch("subprocess.run", return_value=mock_result):
            cases = [
                TestCase(name="case1", inputs={"earned_income": 30000}, expected={}),
            ]
            results = v.batch_validate(cases, "eitc", 2023)
            assert len(results) == 1

    def test_batch_empty_line_in_output(self):
        """Empty lines in batch output are skipped."""
        v = TaxsimValidator(mode="web", max_retries=1)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "taxsimid,year,fiitax,v25\n1,2023,5000.00,600.00\n\n2,2023,3000.00,400.00"
        with patch("subprocess.run", return_value=mock_result):
            cases = [
                TestCase(name="case1", inputs={"earned_income": 30000}, expected={}),
                TestCase(name="case2", inputs={"earned_income": 20000}, expected={}),
            ]
            results = v.batch_validate(cases, "eitc", 2023)
            assert len(results) == 2
            assert results[0].calculated_value == 600.0
            assert results[1].calculated_value == 400.0

    def test_batch_variable_not_in_output(self):
        """When variable column not found in batch output."""
        v = TaxsimValidator(mode="web", max_retries=1)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "taxsimid,year,fiitax\n1,2023,5000.00"
        with patch("subprocess.run", return_value=mock_result):
            cases = [
                TestCase(name="case1", inputs={"earned_income": 30000}, expected={}),
            ]
            results = v.batch_validate(cases, "eitc", 2023)
            assert len(results) == 1
            assert not results[0].success

    def test_batch_direct_var_name_match(self):
        """When var_lower is directly in the row dict (fallback lookup)."""
        v = TaxsimValidator(mode="web", max_retries=1)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "taxsimid,year,eitc\n1,2023,600.00"
        with patch("subprocess.run", return_value=mock_result):
            cases = [
                TestCase(name="case1", inputs={"earned_income": 30000}, expected={}),
            ]
            results = v.batch_validate(cases, "eitc", 2023)
            assert len(results) == 1
            assert results[0].calculated_value == 600.0

    def test_batch_invalid_output_too_few_lines(self):
        """When batch output has fewer than 2 lines (only header or empty)."""
        v = TaxsimValidator(mode="web", max_retries=1)
        # Return only a header line, no data rows
        with patch.object(v, "_execute_web", return_value="taxsimid,year,fiitax"):
            cases = [
                TestCase(name="case1", inputs={"earned_income": 30000}, expected={}),
            ]
            results = v.batch_validate(cases, "eitc", 2023)
            # Should have error results for all cases
            assert len(results) == 1
            assert not results[0].success
            assert "batch execution failed" in results[0].error


class TestExecuteLocalDarwinPath:
    def test_darwin_homebrew_paths_added(self, tmp_path):
        """Test that homebrew paths are added to PATH on macOS when missing."""
        exe = tmp_path / "taxsim35"
        exe.write_text("fake")
        v = TaxsimValidator.__new__(TaxsimValidator)
        v.taxsim_path = exe
        v.mode = "local"

        mock_result = MagicMock()
        mock_result.returncode = 0

        import tempfile

        # Use an environment with no homebrew paths
        env_no_brew = {"PATH": "/usr/bin:/bin"}
        with (
            patch("platform.system", return_value="Darwin"),
            patch("os.environ", env_no_brew),
            patch("subprocess.run", return_value=mock_result) as mock_run,
            patch("os.close"),
            patch("os.chmod"),
            patch("os.path.exists", return_value=True),
            patch("os.unlink"),
            patch.object(tempfile, "mkstemp", return_value=(5, "/tmp/output.csv")),
        ):
            mock_open = MagicMock()
            mock_file = MagicMock()
            mock_file.read.return_value = "taxsimid,year,fiitax\n1,2023,5000"
            mock_open.return_value.__enter__ = MagicMock(return_value=mock_file)
            mock_open.return_value.__exit__ = MagicMock(return_value=False)
            with patch("builtins.open", mock_open):
                v._execute_local("/tmp/input.csv")
                # Verify homebrew paths were added
                call_env = mock_run.call_args[1].get("env", {})
                assert "/opt/homebrew/bin" in call_env.get("PATH", "")
                assert "/usr/local/bin" in call_env.get("PATH", "")


class TestBuildTaxsimInput:
    def test_state_as_integer(self):
        v = TaxsimValidator()
        tc = TestCase(name="test", inputs={"earned_income": 30000, "state": 6}, expected={})
        ti = v._build_taxsim_input(tc, 2023)
        assert ti["state"] == 6

    def test_state_as_string(self):
        v = TaxsimValidator()
        tc = TestCase(name="test", inputs={"earned_income": 30000, "state": "CA"}, expected={})
        ti = v._build_taxsim_input(tc, 2023)
        assert isinstance(ti["state"], int)


class TestResolveTaxsimPathEdgeCases:
    def test_linux_platform(self):
        """Test Linux platform detection."""
        v = TaxsimValidator.__new__(TaxsimValidator)
        with patch("platform.system", return_value="Linux"), patch("pathlib.Path.exists", return_value=True):
            result = v._resolve_taxsim_path(None)
            assert "unix" in str(result).lower() or result.exists()

    def test_windows_platform(self):
        """Test Windows platform detection."""
        v = TaxsimValidator.__new__(TaxsimValidator)
        with patch("platform.system", return_value="Windows"), patch("pathlib.Path.exists", return_value=True):
            result = v._resolve_taxsim_path(None)
            assert "windows" in str(result).lower() or result.exists()

    def test_unsupported_platform(self):
        """Test unsupported OS raises error."""
        v = TaxsimValidator.__new__(TaxsimValidator)
        with patch("platform.system", return_value="FreeBSD"), pytest.raises(OSError, match="Unsupported"):
            v._resolve_taxsim_path(None)

    def test_not_found_anywhere(self):
        """Test FileNotFoundError when exe not found."""
        v = TaxsimValidator.__new__(TaxsimValidator)
        with (
            patch("platform.system", return_value="Darwin"),
            patch("pathlib.Path.exists", return_value=False),
            pytest.raises(FileNotFoundError, match="not found"),
        ):
            v._resolve_taxsim_path(None)
