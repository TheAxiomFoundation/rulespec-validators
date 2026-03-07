"""Tests for CLI module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from cosilico_validators.cli import cli, display_results, display_summary
from cosilico_validators.consensus.engine import ConsensusLevel
from cosilico_validators.validators.base import TestCase, ValidatorType


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def sample_test_file(tmp_path):
    data = [
        {
            "name": "test1",
            "inputs": {"earned_income": 20000, "filing_status": "SINGLE"},
            "expected": {"eitc": 560},
        }
    ]
    path = tmp_path / "tests.json"
    path.write_text(json.dumps(data))
    return str(path)


@pytest.fixture
def sample_yaml_test_file(tmp_path):
    import yaml

    data = [
        {
            "name": "test1",
            "inputs": {"earned_income": 20000},
            "expected": {"eitc": 560},
        }
    ]
    path = tmp_path / "tests.yaml"
    path.write_text(yaml.dump(data))
    return str(path)


@pytest.fixture
def sample_dict_test_file(tmp_path):
    data = {
        "test_cases": [
            {
                "name": "test1",
                "inputs": {"earned_income": 20000},
                "expected": {"eitc": 560},
            }
        ]
    }
    path = tmp_path / "tests.json"
    path.write_text(json.dumps(data))
    return str(path)


@pytest.fixture
def sample_results_file(tmp_path):
    data = [
        {
            "test_case": "test1",
            "potential_bugs": [
                {
                    "validator": "PolicyEngine",
                    "test_case": "test1",
                    "expected": 560,
                    "actual": 600,
                    "difference": 40,
                    "citation": "26 USC 32",
                    "inputs": {"earned_income": 20000},
                    "claude_confidence": 0.95,
                }
            ],
        }
    ]
    path = tmp_path / "results.json"
    path.write_text(json.dumps(data))
    return str(path)


@pytest.fixture
def empty_results_file(tmp_path):
    data = [{"test_case": "test1", "potential_bugs": []}]
    path = tmp_path / "results.json"
    path.write_text(json.dumps(data))
    return str(path)


def _make_mock_validation_result():
    """Create a mock ValidationResult with all the right attributes."""
    mock_result = MagicMock()
    mock_result.test_case = TestCase(name="test1", inputs={}, expected={"eitc": 560})
    mock_result.variable = "eitc"
    mock_result.expected_value = 560
    mock_result.consensus_value = 560.0
    mock_result.consensus_level = ConsensusLevel.FULL_AGREEMENT
    mock_result.reward_signal = 1.0
    mock_result.confidence = 0.95
    mock_result.matches_expected = True
    mock_result.potential_bugs = []
    mock_result.validator_results = {}
    return mock_result


def _make_harness_result(with_alignment=False, with_issues=False, with_review=False):
    """Create a real HarnessResult for CLI testing."""
    from cosilico_validators.harness import (
        AlignmentResult,
        CoverageResult,
        HarnessResult,
        QualityIssue,
        QualityResult,
        ReviewResult,
        VariableAlignment,
    )

    by_variable = {}
    if with_alignment:
        by_variable = {
            "eitc": VariableAlignment(variable="eitc", section="26/32", policyengine=0.95),
        }
    alignment = AlignmentResult(overall_rate=0.95, by_variable=by_variable)
    coverage = CoverageResult(implemented=5, total=13)
    issues = []
    if with_issues:
        issues = [
            QualityIssue(
                file="test.rac",
                line=i,
                category="schema",
                severity="error" if i == 1 else "warning",
                message=f"issue {i}",
            )
            for i in range(1, 7)
        ]
    quality = QualityResult(
        test_coverage=0.5,
        no_literals_pass=True,
        all_imports_valid=True,
        all_dtypes_valid=True,
        issues=issues,
    )
    review = None
    if with_review:
        review = ReviewResult(
            overall_score=7.0,
            accuracy=7.0,
            completeness=7.0,
            parameterization=7.0,
            test_quality=7.0,
            feedback="Looks good",
            reviewed_files=["test.rac"],
        )
    return HarnessResult(
        timestamp="2024-01-01T00:00:00",
        git_commit="abc1234",
        alignment=alignment,
        coverage=coverage,
        quality=quality,
        review=review,
    )


class TestLoadValidators:
    def test_load_both(self):
        from cosilico_validators.cli import load_validators

        with (
            patch("cosilico_validators.cli.TaxsimValidator", create=True),
            patch("cosilico_validators.cli.PolicyEngineValidator", create=True),
            patch("cosilico_validators.validators.taxsim.TaxsimValidator"),
            patch("cosilico_validators.validators.policyengine.PolicyEngineValidator"),
        ):
            # The function does lazy imports, so we patch the import mechanism
            validators = load_validators(include_policyengine=True, include_taxsim=True)
            assert isinstance(validators, list)

    def test_load_taxsim_only(self):
        from cosilico_validators.cli import load_validators

        with patch("cosilico_validators.validators.taxsim.TaxsimValidator"):
            validators = load_validators(include_policyengine=False, include_taxsim=True)
            assert isinstance(validators, list)
            assert len(validators) >= 1

    def test_load_policyengine_import_error(self):
        from cosilico_validators.cli import load_validators

        with patch("cosilico_validators.validators.taxsim.TaxsimValidator"):
            # Simulate PE import failure
            import builtins

            original_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if "policyengine" in name:
                    raise ImportError("no PE")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                validators = load_validators(include_policyengine=True, include_taxsim=True)
                assert isinstance(validators, list)

    def test_load_neither(self):
        from cosilico_validators.cli import load_validators

        validators = load_validators(include_policyengine=False, include_taxsim=False)
        assert validators == []


class TestCliGroup:
    def test_cli_help(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Multi-system" in result.output


class TestValidateCommand:
    def _setup_mocks(self):
        mock_result = _make_mock_validation_result()
        return mock_result

    def test_validate_json(self, runner, sample_test_file):
        mock_result = self._setup_mocks()
        with (
            patch("cosilico_validators.cli.load_validators") as mock_lv,
            patch("cosilico_validators.cli.ConsensusEngine") as mock_ce,
        ):
            mock_lv.return_value = [MagicMock()]
            mock_engine = MagicMock()
            mock_engine.validate.return_value = mock_result
            mock_ce.return_value = mock_engine
            result = runner.invoke(
                cli,
                [
                    "validate",
                    sample_test_file,
                    "-v",
                    "eitc",
                    "--no-policyengine",
                ],
            )
            assert result.exit_code == 0

    def test_validate_yaml(self, runner, sample_yaml_test_file):
        mock_result = self._setup_mocks()
        with (
            patch("cosilico_validators.cli.load_validators") as mock_lv,
            patch("cosilico_validators.cli.ConsensusEngine") as mock_ce,
        ):
            mock_lv.return_value = [MagicMock()]
            mock_engine = MagicMock()
            mock_engine.validate.return_value = mock_result
            mock_ce.return_value = mock_engine
            result = runner.invoke(
                cli,
                [
                    "validate",
                    sample_yaml_test_file,
                    "-v",
                    "eitc",
                    "--no-policyengine",
                ],
            )
            assert result.exit_code == 0

    def test_validate_dict_format(self, runner, sample_dict_test_file):
        mock_result = self._setup_mocks()
        with (
            patch("cosilico_validators.cli.load_validators") as mock_lv,
            patch("cosilico_validators.cli.ConsensusEngine") as mock_ce,
        ):
            mock_lv.return_value = [MagicMock()]
            mock_engine = MagicMock()
            mock_engine.validate.return_value = mock_result
            mock_ce.return_value = mock_engine
            result = runner.invoke(
                cli,
                [
                    "validate",
                    sample_dict_test_file,
                    "-v",
                    "eitc",
                    "--no-policyengine",
                ],
            )
            assert result.exit_code == 0

    def test_validate_unsupported_format(self, runner, tmp_path):
        path = tmp_path / "tests.txt"
        path.write_text("test")
        result = runner.invoke(cli, ["validate", str(path), "-v", "eitc"])
        assert result.exit_code != 0

    def test_validate_empty_file(self, runner, tmp_path):
        path = tmp_path / "tests.json"
        path.write_text("[]")
        result = runner.invoke(cli, ["validate", str(path), "-v", "eitc"])
        assert result.exit_code != 0

    def test_validate_no_validators(self, runner, sample_test_file):
        with patch("cosilico_validators.cli.load_validators") as mock_lv:
            mock_lv.return_value = []
            result = runner.invoke(
                cli,
                [
                    "validate",
                    sample_test_file,
                    "-v",
                    "eitc",
                    "--no-policyengine",
                    "--no-taxsim",
                ],
            )
            assert result.exit_code != 0

    def test_validate_with_output(self, runner, sample_test_file, tmp_path):
        mock_result = self._setup_mocks()
        output_path = str(tmp_path / "output.json")
        with (
            patch("cosilico_validators.cli.load_validators") as mock_lv,
            patch("cosilico_validators.cli.ConsensusEngine") as mock_ce,
        ):
            mock_lv.return_value = [MagicMock()]
            mock_engine = MagicMock()
            mock_engine.validate.return_value = mock_result
            mock_ce.return_value = mock_engine
            result = runner.invoke(
                cli,
                [
                    "validate",
                    sample_test_file,
                    "-v",
                    "eitc",
                    "--no-policyengine",
                    "-o",
                    output_path,
                ],
            )
            assert result.exit_code == 0
            assert Path(output_path).exists()


class TestDisplayResults:
    def test_display_results_no_bugs(self):
        mock_result = _make_mock_validation_result()
        display_results([mock_result])

    def test_display_results_with_bugs(self):
        mock_result = _make_mock_validation_result()
        mock_result.consensus_level = ConsensusLevel.DISAGREEMENT
        mock_result.matches_expected = False
        mock_result.potential_bugs = [{"validator": "PE", "expected": 560, "actual": 600, "difference": 40}]
        display_results([mock_result])

    def test_display_results_null_consensus(self):
        mock_result = _make_mock_validation_result()
        mock_result.consensus_value = None
        mock_result.matches_expected = False
        display_results([mock_result])


class TestDisplaySummary:
    def test_display_summary(self):
        mock_result = _make_mock_validation_result()
        display_summary([mock_result])


class TestValidatorsCommand:
    def test_list_validators(self, runner):
        with patch("cosilico_validators.cli.load_validators") as mock_lv:
            mock_v = MagicMock()
            mock_v.name = "TestValidator"
            mock_v.validator_type = ValidatorType.REFERENCE
            mock_v.supported_variables = {"eitc", "ctc"}
            mock_v.supports_variable.return_value = True
            mock_lv.return_value = [mock_v]
            result = runner.invoke(cli, ["validators"])
            assert result.exit_code == 0

    def test_list_validators_with_variable_filter(self, runner):
        with patch("cosilico_validators.cli.load_validators") as mock_lv:
            mock_v = MagicMock()
            mock_v.name = "TestValidator"
            mock_v.validator_type = ValidatorType.REFERENCE
            mock_v.supported_variables = {"eitc", "ctc"}
            mock_v.supports_variable.return_value = True
            mock_lv.return_value = [mock_v]
            result = runner.invoke(cli, ["validators", "-v", "eitc"])
            assert result.exit_code == 0

    def test_list_validators_with_many_variables(self, runner):
        with patch("cosilico_validators.cli.load_validators") as mock_lv:
            mock_v = MagicMock()
            mock_v.name = "TestValidator"
            mock_v.validator_type = ValidatorType.REFERENCE
            mock_v.supported_variables = {f"var{i}" for i in range(20)}
            mock_v.supports_variable.return_value = True
            mock_lv.return_value = [mock_v]
            result = runner.invoke(cli, ["validators"])
            assert result.exit_code == 0
            assert "more" in result.output


class TestFileIssuesCommand:
    def test_file_issues_with_bugs(self, runner, sample_results_file):
        result = runner.invoke(cli, ["file-issues", sample_results_file, "--dry-run"])
        assert result.exit_code == 0

    def test_file_issues_no_bugs(self, runner, empty_results_file):
        result = runner.invoke(cli, ["file-issues", empty_results_file])
        assert result.exit_code == 0
        assert "No potential bugs" in result.output

    def test_file_issues_with_repo_not_dry_run(self, runner, sample_results_file):
        result = runner.invoke(
            cli,
            [
                "file-issues",
                sample_results_file,
                "--repo",
                "PolicyEngine/policyengine-us",
            ],
        )
        assert result.exit_code == 0


class TestCompareAlignedCommand:
    def test_compare_aligned_import_error(self, runner):
        with patch("cosilico_validators.comparison.run_aligned_comparison", side_effect=ImportError("no PE")):
            result = runner.invoke(cli, ["compare-aligned"])
            assert result.exit_code != 0

    def test_compare_aligned_with_output(self, runner, tmp_path):
        mock_dashboard = {
            "variables": [],
            "summary": {"overall_match_rate": 0, "total_records": 0},
        }
        output_path = str(tmp_path / "output.json")
        with patch("cosilico_validators.comparison.run_aligned_comparison", return_value=mock_dashboard):
            result = runner.invoke(cli, ["compare-aligned", "-o", output_path])
            assert result.exit_code == 0
            assert Path(output_path).exists()

    def test_compare_aligned_with_variables(self, runner):
        """Test compare-aligned table rendering (lines 336-341)."""
        mock_dashboard = {
            "variables": [
                {
                    "variable": "eitc",
                    "match_rate": 0.95,
                    "mean_absolute_error": 50,
                    "cosilico_weighted_total": 60e9,
                    "policyengine_weighted_total": 62e9,
                    "difference_billions": -2.0,
                },
                {
                    "variable": "income_tax",
                    "match_rate": 0.50,
                    "mean_absolute_error": 500,
                    "cosilico_weighted_total": 1000e9,
                    "policyengine_weighted_total": 900e9,
                    "difference_billions": 100.0,
                },
            ],
            "summary": {"overall_match_rate": 0.72, "total_records": 100000},
        }
        with patch("cosilico_validators.comparison.run_aligned_comparison", return_value=mock_dashboard):
            result = runner.invoke(cli, ["compare-aligned"])
            assert result.exit_code == 0


class TestCompareCommand:
    def test_compare(self, runner):
        mock_dashboard = {
            "variables": [{"variable": "eitc", "match_rate": 0.95, "mean_absolute_error": 50, "n_records": 100000}],
            "summary": {"overall_match_rate": 0.95, "total_records": 100000},
        }
        with patch("cosilico_validators.comparison.run_full_comparison", return_value=mock_dashboard):
            result = runner.invoke(cli, ["compare"])
            assert result.exit_code == 0

    def test_compare_with_error_variable(self, runner):
        mock_dashboard = {
            "variables": [{"variable": "eitc", "error": "failed"}],
            "summary": {"overall_match_rate": 0, "total_records": 0},
        }
        with patch("cosilico_validators.comparison.run_full_comparison", return_value=mock_dashboard):
            result = runner.invoke(cli, ["compare"])
            assert result.exit_code == 0

    def test_compare_import_error(self, runner):
        with patch("cosilico_validators.comparison.run_full_comparison", side_effect=ImportError("no PE")):
            result = runner.invoke(cli, ["compare"])
            assert result.exit_code != 0

    def test_compare_with_output(self, runner, tmp_path):
        mock_dashboard = {
            "variables": [],
            "summary": {"overall_match_rate": 0, "total_records": 0},
        }
        output_path = str(tmp_path / "output.json")
        with patch("cosilico_validators.comparison.run_full_comparison", return_value=mock_dashboard):
            result = runner.invoke(cli, ["compare", "-o", output_path])
            assert result.exit_code == 0
            assert Path(output_path).exists()


class TestDashboardCommand:
    def test_dashboard(self, runner):
        mock_data = {
            "sections": [
                {
                    "variable": "eitc",
                    "section": "26/32",
                    "summary": {"matchRate": 0.95, "meanAbsoluteError": 50},
                }
            ],
            "overall": {
                "matchRate": 0.95,
                "totalHouseholds": 100000,
                "meanAbsoluteError": 50,
            },
        }
        with patch("cosilico_validators.dashboard_export.run_export", return_value=mock_data):
            result = runner.invoke(cli, ["dashboard"])
            assert result.exit_code == 0

    def test_dashboard_import_error(self, runner):
        with patch("cosilico_validators.dashboard_export.run_export", side_effect=ImportError("no PE")):
            result = runner.invoke(cli, ["dashboard"])
            assert result.exit_code != 0

    def test_dashboard_with_output(self, runner, tmp_path):
        mock_data = {
            "sections": [],
            "overall": {"matchRate": 0, "totalHouseholds": 0, "meanAbsoluteError": 0},
        }
        output_path = str(tmp_path / "output.json")
        with patch("cosilico_validators.dashboard_export.run_export", return_value=mock_data):
            result = runner.invoke(cli, ["dashboard", "-o", output_path])
            assert result.exit_code == 0


class TestHarnessCommands:
    def test_harness_run(self, runner):
        result_obj = _make_harness_result()
        with (
            patch("cosilico_validators.harness.runner.run_harness", return_value=result_obj),
            patch("cosilico_validators.harness.scorecard.generate_compact_scorecard", return_value="test scorecard"),
        ):
            result = runner.invoke(cli, ["harness", "run"])
            assert result.exit_code == 0

    def test_harness_run_with_alignment(self, runner):
        result_obj = _make_harness_result(with_alignment=True, with_issues=True)
        with (
            patch("cosilico_validators.harness.runner.run_harness", return_value=result_obj),
            patch("cosilico_validators.harness.scorecard.generate_compact_scorecard", return_value="test"),
        ):
            result = runner.invoke(cli, ["harness", "run"])
            assert result.exit_code == 0

    def test_harness_run_with_output(self, runner, tmp_path):
        result_obj = _make_harness_result()
        output_path = str(tmp_path / "output.json")
        with (
            patch("cosilico_validators.harness.runner.run_harness", return_value=result_obj),
            patch("cosilico_validators.harness.scorecard.generate_compact_scorecard", return_value="test"),
            patch("cosilico_validators.harness.checkpoint.save_checkpoint"),
        ):
            result = runner.invoke(cli, ["harness", "run", "-o", output_path])
            assert result.exit_code == 0

    def test_harness_run_with_baseline(self, runner, tmp_path):
        from cosilico_validators.harness import Checkpoint

        result_obj = _make_harness_result()
        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text("{}")
        mock_cp = Checkpoint(
            timestamp="2024-01-01",
            git_commit="def5678",
            scores={"alignment": 0.9, "coverage": 0.5, "quality": 0.7, "review": 0.0},
            details={},
        )
        with (
            patch("cosilico_validators.harness.runner.run_harness", return_value=result_obj),
            patch("cosilico_validators.harness.scorecard.generate_compact_scorecard", return_value="test"),
            patch("cosilico_validators.harness.checkpoint.load_checkpoint", return_value=mock_cp),
        ):
            result = runner.invoke(cli, ["harness", "run", "-b", str(baseline_path)])
            assert result.exit_code == 0

    def test_harness_run_only_quality(self, runner):
        result_obj = _make_harness_result()
        with (
            patch("cosilico_validators.harness.runner.run_harness", return_value=result_obj),
            patch("cosilico_validators.harness.scorecard.generate_compact_scorecard", return_value="test"),
        ):
            result = runner.invoke(cli, ["harness", "run", "--only", "quality"])
            assert result.exit_code == 0

    def test_harness_checkpoint_save(self, runner, tmp_path):
        result_obj = _make_harness_result()
        output_path = str(tmp_path / "checkpoint.json")
        with (
            patch("cosilico_validators.harness.runner.run_harness", return_value=result_obj),
            patch("cosilico_validators.harness.checkpoint.save_checkpoint"),
        ):
            result = runner.invoke(cli, ["harness", "checkpoint", "--save", output_path])
            assert result.exit_code == 0

    def test_harness_checkpoint_named(self, runner):
        result_obj = _make_harness_result()
        with (
            patch("cosilico_validators.harness.runner.run_harness", return_value=result_obj),
            patch("cosilico_validators.harness.checkpoint.save_baseline", return_value=Path("/tmp/test.json")),
        ):
            result = runner.invoke(cli, ["harness", "checkpoint", "--name", "test"])
            assert result.exit_code == 0

    def test_harness_compare(self, runner, tmp_path):
        from cosilico_validators.harness import Checkpoint

        result_obj = _make_harness_result()
        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text("{}")
        mock_cp = Checkpoint(
            timestamp="2024-01-01",
            git_commit="def5678",
            scores={"alignment": 0.9, "coverage": 0.5, "quality": 0.7, "review": 0.0},
            details={},
        )
        with (
            patch("cosilico_validators.harness.runner.run_harness", return_value=result_obj),
            patch("cosilico_validators.harness.checkpoint.load_checkpoint", return_value=mock_cp),
            patch("cosilico_validators.harness.checkpoint.compare_checkpoints") as mock_cc,
        ):
            from cosilico_validators.harness import Delta

            mock_cc.return_value = Delta(before=mock_cp, after=Checkpoint.from_result(result_obj))
            result = runner.invoke(cli, ["harness", "compare", "-b", str(baseline_path)])
            assert result.exit_code == 0

    def test_harness_compare_no_baseline(self, runner, tmp_path):
        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text("{}")
        with patch("cosilico_validators.harness.checkpoint.load_checkpoint", return_value=None):
            result = runner.invoke(cli, ["harness", "compare", "-b", str(baseline_path)])
            assert result.exit_code != 0

    def test_harness_compare_with_current_file(self, runner, tmp_path):
        from cosilico_validators.harness import Checkpoint, Delta

        baseline_path = tmp_path / "baseline.json"
        current_path = tmp_path / "current.json"
        for p in [baseline_path, current_path]:
            p.write_text("{}")
        mock_cp = Checkpoint(
            timestamp="2024-01-01",
            git_commit="abc",
            scores={"alignment": 0.9, "coverage": 0.5, "quality": 0.7, "review": 0.0},
            details={},
        )
        with (
            patch("cosilico_validators.harness.checkpoint.load_checkpoint", return_value=mock_cp),
            patch("cosilico_validators.harness.checkpoint.compare_checkpoints") as mock_cc,
        ):
            mock_cc.return_value = Delta(before=mock_cp, after=mock_cp)
            result = runner.invoke(cli, ["harness", "compare", "-b", str(baseline_path), "-c", str(current_path)])
            assert result.exit_code == 0

    def test_harness_compare_current_load_fails(self, runner, tmp_path):
        from cosilico_validators.harness import Checkpoint

        baseline_path = tmp_path / "baseline.json"
        current_path = tmp_path / "current.json"
        for p in [baseline_path, current_path]:
            p.write_text("{}")
        mock_cp = Checkpoint(timestamp="2024-01-01", git_commit="abc", scores={"alignment": 0.9}, details={})
        with patch("cosilico_validators.harness.checkpoint.load_checkpoint", side_effect=[mock_cp, None]):
            result = runner.invoke(cli, ["harness", "compare", "-b", str(baseline_path), "-c", str(current_path)])
            assert result.exit_code != 0

    def test_harness_scorecard(self, runner):
        result_obj = _make_harness_result()
        with (
            patch("cosilico_validators.harness.runner.run_harness", return_value=result_obj),
            patch("cosilico_validators.harness.scorecard.generate_scorecard", return_value="# Scorecard"),
        ):
            result = runner.invoke(cli, ["harness", "scorecard"])
            assert result.exit_code == 0

    def test_harness_scorecard_with_output(self, runner, tmp_path):
        result_obj = _make_harness_result()
        output_path = str(tmp_path / "scorecard.md")
        with (
            patch("cosilico_validators.harness.runner.run_harness", return_value=result_obj),
            patch("cosilico_validators.harness.scorecard.generate_scorecard", return_value="# Scorecard"),
        ):
            result = runner.invoke(cli, ["harness", "scorecard", "-o", output_path])
            assert result.exit_code == 0
            assert Path(output_path).exists()

    def test_harness_scorecard_with_after(self, runner, tmp_path):
        """Test scorecard with --after flag (lines 691-696)."""
        from cosilico_validators.harness import Checkpoint

        result_obj = _make_harness_result()
        after_path = tmp_path / "after.json"
        after_path.write_text("{}")
        mock_cp = Checkpoint(
            timestamp="2024-01-01",
            git_commit="abc",
            scores={"alignment": 0.9, "coverage": 0.5, "quality": 0.7, "review": 0.0},
            details={},
        )
        with (
            patch("cosilico_validators.harness.runner.run_harness", return_value=result_obj),
            patch("cosilico_validators.harness.checkpoint.load_checkpoint", return_value=mock_cp),
            patch("cosilico_validators.harness.scorecard.generate_scorecard", return_value="# Scorecard"),
        ):
            result = runner.invoke(cli, ["harness", "scorecard", "-a", str(after_path)])
            assert result.exit_code == 0

    def test_harness_scorecard_after_load_fails(self, runner, tmp_path):
        """Test scorecard with --after that fails to load."""
        result_obj = _make_harness_result()
        after_path = tmp_path / "after.json"
        after_path.write_text("{}")
        with (
            patch("cosilico_validators.harness.runner.run_harness", return_value=result_obj),
            patch("cosilico_validators.harness.checkpoint.load_checkpoint", return_value=None),
        ):
            result = runner.invoke(cli, ["harness", "scorecard", "-a", str(after_path)])
            assert result.exit_code != 0

    def test_harness_scorecard_with_before(self, runner, tmp_path):
        from cosilico_validators.harness import Checkpoint

        result_obj = _make_harness_result()
        before_path = tmp_path / "before.json"
        before_path.write_text("{}")
        mock_cp = Checkpoint(
            timestamp="2024-01-01",
            git_commit="abc",
            scores={"alignment": 0.9, "coverage": 0.5, "quality": 0.7, "review": 0.0},
            details={},
        )
        with (
            patch("cosilico_validators.harness.runner.run_harness", return_value=result_obj),
            patch("cosilico_validators.harness.checkpoint.load_checkpoint", return_value=mock_cp),
            patch("cosilico_validators.harness.scorecard.generate_scorecard", return_value="# Scorecard"),
        ):
            result = runner.invoke(cli, ["harness", "scorecard", "-b", str(before_path)])
            assert result.exit_code == 0
