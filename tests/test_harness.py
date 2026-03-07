"""Tests for harness module - __init__.py, checkpoint.py, runner.py, scorecard.py."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cosilico_validators.harness import (
    AlignmentResult,
    Checkpoint,
    CoverageResult,
    Delta,
    HarnessResult,
    QualityIssue,
    QualityResult,
    ReviewResult,
    VariableAlignment,
)
from cosilico_validators.harness.checkpoint import (
    compare_checkpoints,
    create_empty_checkpoint,
    get_baseline_path,
    get_git_commit,
    load_baseline,
    load_checkpoint,
    save_baseline,
    save_checkpoint,
)
from cosilico_validators.harness.runner import (
    VARIABLES,
    ValidationHarness,
    run_harness,
)
from cosilico_validators.harness.scorecard import (
    format_delta,
    format_percentage,
    generate_compact_scorecard,
    generate_scorecard,
)

# ============================================================================
# harness/__init__.py dataclass tests
# ============================================================================


class TestVariableAlignment:
    def test_consensus_all_rates(self):
        va = VariableAlignment(variable="eitc", section="26/32", policyengine=0.95, taxsim=0.90, taxcalc=0.85)
        assert va.consensus == pytest.approx(0.9)

    def test_consensus_some_rates(self):
        va = VariableAlignment(variable="eitc", section="26/32", policyengine=0.95)
        assert va.consensus == pytest.approx(0.95)

    def test_consensus_no_rates(self):
        va = VariableAlignment(variable="eitc", section="26/32")
        assert va.consensus == 0.0

    def test_consensus_with_prd(self):
        va = VariableAlignment(variable="eitc", section="26/32", prd=0.80)
        assert va.consensus == pytest.approx(0.80)


class TestCoverageResult:
    def test_percentage(self):
        cr = CoverageResult(implemented=5, total=10)
        assert cr.percentage == pytest.approx(0.5)

    def test_percentage_zero_total(self):
        cr = CoverageResult(implemented=0, total=0)
        assert cr.percentage == 0.0


class TestQualityResult:
    def test_overall_score_all_pass(self):
        qr = QualityResult(test_coverage=1.0, no_literals_pass=True, all_imports_valid=True, all_dtypes_valid=True)
        assert qr.overall_score == 100.0

    def test_overall_score_all_fail(self):
        qr = QualityResult(
            test_coverage=0.0,
            no_literals_pass=False,
            all_imports_valid=False,
            all_dtypes_valid=False,
            all_grounded=False,
        )
        assert qr.overall_score == 0.0

    def test_overall_score_partial(self):
        qr = QualityResult(test_coverage=0.5, no_literals_pass=True, all_imports_valid=False, all_dtypes_valid=True)
        # 20 (literals) + 0 (imports) + 20 (dtypes) + 20 (grounded default True) + 0.5*20 (coverage) = 70
        assert qr.overall_score == pytest.approx(70.0)


class TestHarnessResult:
    def _make_result(self, with_review=False):
        alignment = AlignmentResult(
            overall_rate=0.95,
            by_variable={
                "eitc": VariableAlignment(variable="eitc", section="26/32", policyengine=0.95),
            },
            by_validator={"policyengine": 0.95},
        )
        coverage = CoverageResult(implemented=5, total=13, by_section={"26/32": (1, 1)})
        quality = QualityResult(
            test_coverage=0.8,
            no_literals_pass=True,
            all_imports_valid=True,
            all_dtypes_valid=True,
            issues=[QualityIssue(file="test.rac", line=1, category="test", severity="warning", message="test")],
        )
        review = None
        if with_review:
            review = ReviewResult(
                overall_score=7.0,
                accuracy=7.0,
                completeness=7.0,
                parameterization=7.0,
                test_quality=7.0,
                feedback="Good",
                reviewed_files=["test.rac"],
            )
        return HarnessResult(
            timestamp="2024-01-01",
            git_commit="abc",
            alignment=alignment,
            coverage=coverage,
            quality=quality,
            review=review,
        )

    def test_to_dict(self):
        result = self._make_result()
        d = result.to_dict()
        assert d["timestamp"] == "2024-01-01"
        assert d["git_commit"] == "abc"
        assert d["alignment"]["overall_rate"] == 0.95
        assert "eitc" in d["alignment"]["by_variable"]
        assert d["coverage"]["implemented"] == 5
        assert d["quality"]["test_coverage"] == 0.8
        assert d["review"] is None

    def test_to_dict_with_review(self):
        result = self._make_result(with_review=True)
        d = result.to_dict()
        assert d["review"]["overall_score"] == 7.0
        assert d["review"]["feedback"] == "Good"
        assert d["review"]["reviewed_files"] == ["test.rac"]


class TestCheckpointDataclass:
    def test_from_result(self):
        alignment = AlignmentResult(overall_rate=0.95)
        coverage = CoverageResult(implemented=5, total=13)
        quality = QualityResult(test_coverage=0.8, no_literals_pass=True, all_imports_valid=True, all_dtypes_valid=True)
        review = ReviewResult(
            overall_score=7.0, accuracy=7.0, completeness=7.0, parameterization=7.0, test_quality=7.0, feedback="Good"
        )
        result = HarnessResult(
            timestamp="2024-01-01",
            git_commit="abc",
            alignment=alignment,
            coverage=coverage,
            quality=quality,
            review=review,
        )
        cp = Checkpoint.from_result(result)
        assert cp.scores["alignment"] == 0.95
        assert cp.scores["coverage"] == pytest.approx(5 / 13)
        assert cp.scores["review"] == pytest.approx(0.7)

    def test_from_result_no_review(self):
        alignment = AlignmentResult(overall_rate=0.95)
        coverage = CoverageResult(implemented=5, total=13)
        quality = QualityResult(test_coverage=0.8, no_literals_pass=True, all_imports_valid=True, all_dtypes_valid=True)
        result = HarnessResult(
            timestamp="2024-01-01", git_commit="abc", alignment=alignment, coverage=coverage, quality=quality
        )
        cp = Checkpoint.from_result(result)
        assert cp.scores["review"] == 0.0


class TestDelta:
    def test_delta_properties(self):
        before = Checkpoint(
            timestamp="t1",
            git_commit="a",
            scores={"alignment": 0.8, "coverage": 0.5, "quality": 0.6, "review": 0.7},
            details={},
        )
        after = Checkpoint(
            timestamp="t2",
            git_commit="b",
            scores={"alignment": 0.9, "coverage": 0.6, "quality": 0.7, "review": 0.8},
            details={},
        )
        delta = Delta(before=before, after=after)
        assert delta.alignment_delta == pytest.approx(0.1)
        assert delta.coverage_delta == pytest.approx(0.1)
        assert delta.quality_delta == pytest.approx(0.1)
        assert delta.review_delta == pytest.approx(0.1)

    def test_has_regression_false(self):
        before = Checkpoint(
            timestamp="t1", git_commit="a", scores={"alignment": 0.8, "coverage": 0.5, "quality": 0.6}, details={}
        )
        after = Checkpoint(
            timestamp="t2", git_commit="b", scores={"alignment": 0.9, "coverage": 0.6, "quality": 0.7}, details={}
        )
        delta = Delta(before=before, after=after)
        assert not delta.has_regression()

    def test_has_regression_true(self):
        before = Checkpoint(
            timestamp="t1", git_commit="a", scores={"alignment": 0.9, "coverage": 0.5, "quality": 0.6}, details={}
        )
        after = Checkpoint(
            timestamp="t2", git_commit="b", scores={"alignment": 0.8, "coverage": 0.6, "quality": 0.7}, details={}
        )
        delta = Delta(before=before, after=after)
        assert delta.has_regression()

    def test_missing_scores_default_zero(self):
        before = Checkpoint(timestamp="t1", git_commit="a", scores={}, details={})
        after = Checkpoint(timestamp="t2", git_commit="b", scores={}, details={})
        delta = Delta(before=before, after=after)
        assert delta.alignment_delta == 0
        assert not delta.has_regression()


# ============================================================================
# harness/checkpoint.py tests
# ============================================================================


class TestGetGitCommit:
    def test_returns_string(self):
        commit = get_git_commit()
        assert isinstance(commit, str)

    def test_returns_unknown_on_error(self):
        with patch("subprocess.run", side_effect=Exception("fail")):
            assert get_git_commit() == "unknown"

    def test_returns_unknown_on_empty(self):
        mock_result = MagicMock()
        mock_result.stdout = ""
        with patch("subprocess.run", return_value=mock_result):
            assert get_git_commit() == "unknown"


class TestSaveLoadCheckpoint:
    def _make_result(self):
        return HarnessResult(
            timestamp="2024-01-01",
            git_commit="abc",
            alignment=AlignmentResult(overall_rate=0.95),
            coverage=CoverageResult(implemented=5, total=13),
            quality=QualityResult(
                test_coverage=0.8, no_literals_pass=True, all_imports_valid=True, all_dtypes_valid=True
            ),
        )

    def test_save_and_load(self, tmp_path):
        result = self._make_result()
        path = tmp_path / "checkpoint.json"
        save_checkpoint(result, path)
        loaded = load_checkpoint(path)
        assert loaded is not None
        assert loaded.git_commit == "abc"
        assert loaded.scores["alignment"] == 0.95

    def test_load_nonexistent(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        assert load_checkpoint(path) is None

    def test_save_creates_parent_dirs(self, tmp_path):
        result = self._make_result()
        path = tmp_path / "sub" / "dir" / "checkpoint.json"
        save_checkpoint(result, path)
        assert path.exists()


class TestCompareCheckpoints:
    def test_compare(self):
        before = Checkpoint(timestamp="t1", git_commit="a", scores={"alignment": 0.8}, details={})
        after = Checkpoint(timestamp="t2", git_commit="b", scores={"alignment": 0.9}, details={})
        delta = compare_checkpoints(before, after)
        assert isinstance(delta, Delta)
        assert delta.alignment_delta == pytest.approx(0.1)


class TestBaselineFunctions:
    def test_get_baseline_path(self):
        path = get_baseline_path("main")
        assert path.name == "main.json"

    def test_save_and_load_baseline(self, tmp_path):
        result = HarnessResult(
            timestamp="2024-01-01",
            git_commit="abc",
            alignment=AlignmentResult(overall_rate=0.95),
            coverage=CoverageResult(implemented=5, total=13),
            quality=QualityResult(
                test_coverage=0.8, no_literals_pass=True, all_imports_valid=True, all_dtypes_valid=True
            ),
        )
        with patch("cosilico_validators.harness.checkpoint.get_baseline_path", return_value=tmp_path / "test.json"):
            path = save_baseline(result, "test")
            assert path.exists()
            loaded = load_baseline("test")
            assert loaded is not None

    def test_create_empty_checkpoint(self):
        cp = create_empty_checkpoint()
        assert cp.git_commit == "none"
        assert cp.scores["alignment"] == 0.0


# ============================================================================
# harness/runner.py tests
# ============================================================================


class TestValidationHarness:
    def test_init_defaults(self):
        h = ValidationHarness()
        assert h.run_alignment is True
        assert h.run_quality is True
        assert h.run_review is False

    def test_run_no_alignment_no_quality(self):
        with patch("cosilico_validators.harness.runner.get_git_commit", return_value="abc"):
            h = ValidationHarness(run_alignment=False, run_quality=False)
            result = h.run_full_validation()
            assert result.alignment.overall_rate == 0.0

    def test_coverage_checks(self, tmp_path):
        statute_root = tmp_path / "statute"
        statute_root.mkdir()
        # Create a .rac file for one section
        section_path = statute_root / "26" / "32"
        section_path.mkdir(parents=True)
        (section_path / "a.rac").write_text("variable eitc:\n  formula: |\n    0")

        with patch("cosilico_validators.harness.runner.get_git_commit", return_value="abc"):
            h = ValidationHarness(statute_root=statute_root, run_alignment=False, run_quality=False)
            result = h.run_full_validation()
            assert result.coverage.total == len(VARIABLES)
            assert result.coverage.implemented >= 1

    def test_agent_review_no_rac_files(self):
        with patch("cosilico_validators.harness.runner.get_git_commit", return_value="abc"):
            h = ValidationHarness(run_alignment=False, run_quality=False, run_review=True)
            result = h.run_full_validation(changed_files=[Path("test.py")])
            assert result.review is None

    def test_agent_review_with_rac_files(self):
        with patch("cosilico_validators.harness.runner.get_git_commit", return_value="abc"):
            h = ValidationHarness(run_alignment=False, run_quality=False, run_review=True)
            result = h.run_full_validation(changed_files=[Path("test.rac")])
            assert result.review is not None
            assert result.review.overall_score == 7.0

    def test_run_with_quality_enabled(self, tmp_path):
        """Test run_full_validation with run_quality=True to cover line 96."""
        statute_root = tmp_path / "statute"
        statute_root.mkdir()
        (statute_root / "test.rac").write_text("variable x:\n  formula: |\n    0\n")

        with patch("cosilico_validators.harness.runner.get_git_commit", return_value="abc"):
            h = ValidationHarness(
                statute_root=statute_root,
                run_alignment=False,
                run_quality=True,
            )
            result = h.run_full_validation()
            assert result.quality is not None
            assert isinstance(result.quality.test_coverage, float)

    def test_alignment_checks_success(self):
        mock_dashboard = {
            "sections": [
                {"variable": "eitc", "summary": {"matchRate": 0.95}},
            ]
        }
        with (
            patch("cosilico_validators.harness.runner.get_git_commit", return_value="abc"),
            patch("cosilico_validators.dashboard_export.run_export", return_value=mock_dashboard),
        ):
            h = ValidationHarness(run_alignment=True, run_quality=False)
            result = h.run_full_validation()
            assert result.alignment.overall_rate > 0

    def test_alignment_checks_failure(self):
        with (
            patch("cosilico_validators.harness.runner.get_git_commit", return_value="abc"),
            patch("cosilico_validators.dashboard_export.run_export", side_effect=Exception("fail")),
        ):
            h = ValidationHarness(run_alignment=True, run_quality=False)
            result = h.run_full_validation()
            assert result.alignment.overall_rate == 0.0


class TestRunHarness:
    def test_run_harness_default(self):
        with patch("cosilico_validators.harness.runner.ValidationHarness") as mock_vh:
            mock_vh.return_value.run_full_validation.return_value = MagicMock()
            result = run_harness()
            assert result is not None

    def test_run_harness_only_quality(self):
        with patch("cosilico_validators.harness.runner.ValidationHarness") as mock_vh:
            mock_vh.return_value.run_full_validation.return_value = MagicMock()
            run_harness(only="quality")
            args = mock_vh.call_args
            assert args[1]["run_quality"] is True
            assert args[1]["run_alignment"] is False


# ============================================================================
# harness/scorecard.py tests
# ============================================================================


class TestFormatDelta:
    def test_positive_percentage(self):
        assert "+" in format_delta(0.05)
        assert "arrow_up" in format_delta(0.05)

    def test_negative_percentage(self):
        assert "-" in format_delta(-0.05)
        assert "arrow_down" in format_delta(-0.05)

    def test_zero_delta(self):
        assert format_delta(0.0) == "-"

    def test_non_percentage(self):
        result = format_delta(0.5, is_percentage=False)
        assert "+0.50" in result


class TestFormatPercentage:
    def test_format(self):
        assert format_percentage(0.95) == "95.0%"
        assert format_percentage(1.0) == "100.0%"
        assert format_percentage(0.0) == "0.0%"


class TestGenerateScorecard:
    def _make_result(self, with_review=False, with_issues=False):
        issues = []
        if with_issues:
            issues = [
                QualityIssue(
                    file="test.rac",
                    line=i,
                    category="test",
                    severity="error" if i <= 5 else "warning",
                    message=f"issue {i}",
                )
                for i in range(1, 12)  # 11 issues to test truncation
            ]
        return HarnessResult(
            timestamp="2024-01-01",
            git_commit="abc",
            alignment=AlignmentResult(
                overall_rate=0.95,
                by_variable={
                    "eitc": VariableAlignment(variable="eitc", section="26/32", policyengine=0.95, taxsim=0.90),
                },
                by_validator={"policyengine": 0.95},
            ),
            coverage=CoverageResult(implemented=5, total=13, by_section={"26/32": (1, 1), "26/24": (2, 3)}),
            quality=QualityResult(
                test_coverage=0.8, no_literals_pass=True, all_imports_valid=True, all_dtypes_valid=True, issues=issues
            ),
            review=ReviewResult(
                overall_score=7.0,
                accuracy=7.0,
                completeness=7.0,
                parameterization=7.0,
                test_quality=7.0,
                feedback="Line 1\nLine 2",
                reviewed_files=["test.rac"],
            )
            if with_review
            else None,
        )

    def test_scorecard_without_baseline(self):
        result = self._make_result()
        sc = generate_scorecard(result)
        assert "Scorecard" in sc
        assert "95.0%" in sc
        assert "5/13" in sc

    def test_scorecard_with_baseline(self):
        result = self._make_result()
        baseline = Checkpoint(
            timestamp="t1",
            git_commit="a",
            scores={"alignment": 0.90, "coverage": 0.3, "quality": 0.5, "review": 0.0},
            details={},
        )
        sc = generate_scorecard(result, baseline)
        assert "Scorecard" in sc
        assert "arrow_up" in sc  # alignment improved

    def test_scorecard_with_review(self):
        result = self._make_result(with_review=True)
        sc = generate_scorecard(result)
        assert "Agent Review" in sc
        assert "7.0/10" in sc
        assert "test.rac" in sc

    def test_scorecard_with_issues(self):
        result = self._make_result(with_issues=True)
        sc = generate_scorecard(result)
        assert "Issues Found" in sc
        assert "more issues" in sc  # truncated at 10

    def test_scorecard_with_baseline_and_review(self):
        result = self._make_result(with_review=True)
        baseline = Checkpoint(
            timestamp="t1",
            git_commit="a",
            scores={"alignment": 0.90, "coverage": 0.3, "quality": 0.5, "review": 0.5},
            details={},
        )
        sc = generate_scorecard(result, baseline)
        assert "Review" in sc

    def test_scorecard_no_alignment_variables(self):
        result = HarnessResult(
            timestamp="2024-01-01",
            git_commit="abc",
            alignment=AlignmentResult(overall_rate=0.0),
            coverage=CoverageResult(implemented=0, total=13),
            quality=QualityResult(
                test_coverage=0.0, no_literals_pass=False, all_imports_valid=False, all_dtypes_valid=False
            ),
        )
        sc = generate_scorecard(result)
        assert "Scorecard" in sc
        assert "FAIL" in sc


class TestGenerateCompactScorecard:
    def test_compact_no_baseline(self):
        result = HarnessResult(
            timestamp="2024-01-01",
            git_commit="abc",
            alignment=AlignmentResult(overall_rate=0.95),
            coverage=CoverageResult(implemented=5, total=13),
            quality=QualityResult(
                test_coverage=0.8, no_literals_pass=True, all_imports_valid=True, all_dtypes_valid=True
            ),
        )
        sc = generate_compact_scorecard(result)
        assert "Alignment: 95.0%" in sc
        assert "Coverage: 5/13" in sc

    def test_compact_with_baseline(self):
        result = HarnessResult(
            timestamp="2024-01-01",
            git_commit="abc",
            alignment=AlignmentResult(overall_rate=0.95),
            coverage=CoverageResult(implemented=5, total=13),
            quality=QualityResult(
                test_coverage=0.8, no_literals_pass=True, all_imports_valid=True, all_dtypes_valid=True
            ),
        )
        baseline = Checkpoint(timestamp="t1", git_commit="a", scores={"alignment": 0.90}, details={})
        sc = generate_compact_scorecard(result, baseline)
        assert "+5.0%" in sc

    def test_compact_with_review(self):
        result = HarnessResult(
            timestamp="2024-01-01",
            git_commit="abc",
            alignment=AlignmentResult(overall_rate=0.95),
            coverage=CoverageResult(implemented=5, total=13),
            quality=QualityResult(
                test_coverage=0.8, no_literals_pass=True, all_imports_valid=True, all_dtypes_valid=True
            ),
            review=ReviewResult(
                overall_score=7.0,
                accuracy=7.0,
                completeness=7.0,
                parameterization=7.0,
                test_quality=7.0,
                feedback="Good",
            ),
        )
        sc = generate_compact_scorecard(result)
        assert "Review: 7.0/10" in sc

    def test_compact_no_delta(self):
        result = HarnessResult(
            timestamp="2024-01-01",
            git_commit="abc",
            alignment=AlignmentResult(overall_rate=0.95),
            coverage=CoverageResult(implemented=5, total=13),
            quality=QualityResult(
                test_coverage=0.8, no_literals_pass=True, all_imports_valid=True, all_dtypes_valid=True
            ),
        )
        baseline = Checkpoint(timestamp="t1", git_commit="a", scores={"alignment": 0.95}, details={})
        sc = generate_compact_scorecard(result, baseline)
        assert "+0.0%" not in sc  # No delta shown for zero change
