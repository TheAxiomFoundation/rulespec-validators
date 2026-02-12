"""Tests for upstream/github module."""

import os
from unittest.mock import MagicMock, patch

from cosilico_validators.upstream import GitHubIssueManager as UpstreamGitHubIssueManager
from cosilico_validators.upstream.github import (
    VALIDATOR_REPOS,
    GitHubIssueManager,
    IssueReport,
)


class TestUpstreamInit:
    def test_exports(self):
        assert UpstreamGitHubIssueManager is GitHubIssueManager


class TestValidatorRepos:
    def test_policyengine(self):
        assert VALIDATOR_REPOS["PolicyEngine"] == "PolicyEngine/policyengine-us"

    def test_taxsim_no_repo(self):
        assert VALIDATOR_REPOS["TAXSIM"] is None


class TestIssueReport:
    def test_creation(self):
        report = IssueReport(
            validator="PolicyEngine",
            test_case="EITC basic test",
            variable="eitc",
            expected=560,
            actual=600,
            difference=40,
            citation="26 USC 32",
            inputs={"earned_income": 20000},
            claude_confidence=0.95,
        )
        assert report.validator == "PolicyEngine"
        assert report.expected == 560

    def test_to_markdown(self):
        report = IssueReport(
            validator="PolicyEngine",
            test_case="EITC basic test",
            variable="eitc",
            expected=560,
            actual=600,
            difference=40,
            citation="26 USC 32",
            inputs={"earned_income": 20000},
            claude_confidence=0.95,
        )
        title, body = report.to_markdown()
        assert "EITC basic test" in title
        assert "eitc" in title
        assert "$560.00" in body
        assert "$600.00" in body
        assert "26 USC 32" in body
        assert "95.0%" in body

    def test_to_markdown_no_citation(self):
        report = IssueReport(
            validator="PE", test_case="test", variable="eitc",
            expected=100, actual=200, difference=100,
            citation=None, inputs={}, claude_confidence=None,
        )
        title, body = report.to_markdown()
        assert "Not provided" in body
        assert "N/A" in body


class TestGitHubIssueManager:
    def test_init_no_token(self):
        with patch.dict(os.environ, {}, clear=True):
            manager = GitHubIssueManager()
            assert manager.token is None

    def test_init_with_token(self):
        manager = GitHubIssueManager(token="test_token")
        assert manager.token == "test_token"

    def test_init_env_token(self):
        with patch.dict(os.environ, {"GITHUB_TOKEN": "env_token"}):
            manager = GitHubIssueManager()
            assert manager.token == "env_token"

    def test_headers_without_token(self):
        manager = GitHubIssueManager()
        headers = manager._headers()
        assert "Accept" in headers
        assert "Authorization" not in headers

    def test_headers_with_token(self):
        manager = GitHubIssueManager(token="test_token")
        headers = manager._headers()
        assert headers["Authorization"] == "Bearer test_token"

    def test_create_issue_report(self):
        manager = GitHubIssueManager()
        bug = {
            "validator": "PolicyEngine",
            "test_case": "test",
            "variable": "eitc",
            "expected": 100,
            "actual": 200,
            "difference": 100,
            "citation": "26 USC 32",
            "inputs": {"x": 1},
            "claude_confidence": 0.95,
        }
        report = manager.create_issue_report(bug)
        assert isinstance(report, IssueReport)
        assert report.validator == "PolicyEngine"

    def test_create_issue_report_missing_fields(self):
        manager = GitHubIssueManager()
        bug = {
            "validator": "PE",
            "test_case": "test",
            "expected": 100,
            "actual": 200,
            "difference": 100,
        }
        report = manager.create_issue_report(bug)
        assert report.variable == "unknown"
        assert report.inputs == {}
        assert report.claude_confidence is None

    def test_file_issue_dry_run(self):
        manager = GitHubIssueManager()
        report = IssueReport(
            validator="PolicyEngine", test_case="test", variable="eitc",
            expected=100, actual=200, difference=100,
            citation="26 USC 32", inputs={}, claude_confidence=0.95,
        )
        result = manager.file_issue(report, dry_run=True)
        assert result["dry_run"] is True
        assert result["repo"] == "PolicyEngine/policyengine-us"

    def test_file_issue_no_repo_for_validator(self):
        manager = GitHubIssueManager()
        report = IssueReport(
            validator="TAXSIM", test_case="test", variable="eitc",
            expected=100, actual=200, difference=100,
            citation=None, inputs={}, claude_confidence=0.95,
        )
        result = manager.file_issue(report)
        assert result["skipped"] is True

    def test_file_issue_no_token(self):
        manager = GitHubIssueManager()
        report = IssueReport(
            validator="PolicyEngine", test_case="test", variable="eitc",
            expected=100, actual=200, difference=100,
            citation=None, inputs={}, claude_confidence=0.95,
        )
        result = manager.file_issue(report)
        assert result["skipped"] is True
        assert "token" in result["error"].lower()

    def test_file_issue_with_override_repo(self):
        manager = GitHubIssueManager()
        report = IssueReport(
            validator="TAXSIM", test_case="test", variable="eitc",
            expected=100, actual=200, difference=100,
            citation=None, inputs={}, claude_confidence=0.95,
        )
        result = manager.file_issue(report, repo="test/repo", dry_run=True)
        assert result["repo"] == "test/repo"

    def test_file_issue_success(self):
        manager = GitHubIssueManager(token="test_token")
        report = IssueReport(
            validator="PolicyEngine", test_case="test", variable="eitc",
            expected=100, actual=200, difference=100,
            citation=None, inputs={}, claude_confidence=0.95,
        )
        mock_dup_response = MagicMock()
        mock_dup_response.status_code = 200
        mock_dup_response.json.return_value = {"total_count": 0}

        mock_create_response = MagicMock()
        mock_create_response.status_code = 201
        mock_create_response.json.return_value = {
            "number": 1, "html_url": "https://github.com/test/1",
        }

        with patch("requests.get", return_value=mock_dup_response), \
             patch("requests.post", return_value=mock_create_response):
            result = manager.file_issue(report)
            assert result["number"] == 1
            assert len(manager.filed_issues) == 1

    def test_file_issue_duplicate(self):
        manager = GitHubIssueManager(token="test_token")
        report = IssueReport(
            validator="PolicyEngine", test_case="test", variable="eitc",
            expected=100, actual=200, difference=100,
            citation=None, inputs={}, claude_confidence=0.95,
        )
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "total_count": 1,
            "items": [{"number": 42, "html_url": "https://github.com/test/42"}],
        }

        with patch("requests.get", return_value=mock_response):
            result = manager.file_issue(report)
            assert result["duplicate"] is True

    def test_file_issue_api_failure(self):
        manager = GitHubIssueManager(token="test_token")
        report = IssueReport(
            validator="PolicyEngine", test_case="test", variable="eitc",
            expected=100, actual=200, difference=100,
            citation=None, inputs={}, claude_confidence=0.95,
        )
        mock_dup = MagicMock()
        mock_dup.status_code = 200
        mock_dup.json.return_value = {"total_count": 0}

        mock_create = MagicMock()
        mock_create.status_code = 422
        mock_create.text = "Validation Failed"

        with patch("requests.get", return_value=mock_dup), \
             patch("requests.post", return_value=mock_create):
            result = manager.file_issue(report)
            assert "error" in result

    def test_file_all_bugs(self):
        manager = GitHubIssueManager()
        bugs = [
            {"validator": "PolicyEngine", "test_case": "t1", "expected": 100,
             "actual": 200, "difference": 100, "claude_confidence": 0.95},
            {"validator": "PolicyEngine", "test_case": "t2", "expected": 300,
             "actual": 400, "difference": 100, "claude_confidence": 0.5},
        ]
        results = manager.file_all_bugs(bugs, dry_run=True)
        assert len(results) == 2
        # Second one should be skipped (confidence below threshold)
        assert results[1]["skipped"] is True
        assert "confidence" in results[1]["reason"].lower()

    def test_file_all_bugs_custom_threshold(self):
        manager = GitHubIssueManager()
        bugs = [
            {"validator": "PolicyEngine", "test_case": "t1", "expected": 100,
             "actual": 200, "difference": 100, "claude_confidence": 0.5},
        ]
        results = manager.file_all_bugs(bugs, dry_run=True, confidence_threshold=0.3)
        assert len(results) == 1
        assert "dry_run" in results[0]

    def test_filed_issues_property(self):
        manager = GitHubIssueManager()
        assert manager.filed_issues == []

    def test_check_duplicate_failure(self):
        manager = GitHubIssueManager(token="test_token")
        mock_response = MagicMock()
        mock_response.status_code = 500
        with patch("requests.get", return_value=mock_response):
            result = manager._check_duplicate("test/repo", "title")
            assert result is None
