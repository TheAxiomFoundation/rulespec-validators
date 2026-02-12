"""Validation harness for Cosilico encoding work.

Provides comprehensive validation across multiple external calculators,
quality checks for .rac files, and PR scorecard generation.
"""

from dataclasses import dataclass, field
from typing import Optional

__all__ = [
    "HarnessResult",
    "AlignmentResult",
    "VariableAlignment",
    "CoverageResult",
    "QualityResult",
    "QualityIssue",
    "ReviewResult",
    "Checkpoint",
    "Delta",
]


@dataclass
class VariableAlignment:
    """Match rates for a single variable across validators."""

    variable: str
    section: str  # USC section (e.g., "26/32")
    policyengine: Optional[float] = None
    taxsim: Optional[float] = None
    taxcalc: Optional[float] = None
    prd: Optional[float] = None

    @property
    def consensus(self) -> float:
        """Average of available validators."""
        rates = [r for r in [self.policyengine, self.taxsim, self.taxcalc, self.prd] if r is not None]
        return sum(rates) / len(rates) if rates else 0.0


@dataclass
class AlignmentResult:
    """Alignment scores across all validators."""

    overall_rate: float  # Weighted average
    by_variable: dict[str, VariableAlignment] = field(default_factory=dict)
    by_validator: dict[str, float] = field(default_factory=dict)


@dataclass
class CoverageResult:
    """Coverage of implemented variables."""

    implemented: int
    total: int
    by_section: dict[str, tuple[int, int]] = field(default_factory=dict)  # (implemented, total)

    @property
    def percentage(self) -> float:
        return self.implemented / self.total if self.total > 0 else 0.0


@dataclass
class QualityIssue:
    """A quality issue found in a .rac file."""

    file: str
    line: Optional[int]
    category: str  # schema, literal, import, test_coverage
    severity: str  # error, warning, info
    message: str


@dataclass
class QualityResult:
    """Quality metrics for .rac files."""

    test_coverage: float  # % variables with tests
    no_literals_pass: bool
    all_imports_valid: bool
    all_dtypes_valid: bool
    issues: list[QualityIssue] = field(default_factory=list)

    @property
    def overall_score(self) -> float:
        """Overall quality score 0-100."""
        score = 0.0
        if self.no_literals_pass:
            score += 25
        if self.all_imports_valid:
            score += 25
        if self.all_dtypes_valid:
            score += 25
        score += self.test_coverage * 25
        return score


@dataclass
class ReviewResult:
    """Agent review scores."""

    overall_score: float  # 1-10
    accuracy: float  # 1-10
    completeness: float  # 1-10
    parameterization: float  # 1-10
    test_quality: float  # 1-10
    feedback: str
    reviewed_files: list[str] = field(default_factory=list)


@dataclass
class HarnessResult:
    """Complete validation harness result."""

    timestamp: str
    git_commit: str
    alignment: AlignmentResult
    coverage: CoverageResult
    quality: QualityResult
    review: Optional[ReviewResult] = None

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "timestamp": self.timestamp,
            "git_commit": self.git_commit,
            "alignment": {
                "overall_rate": self.alignment.overall_rate,
                "by_variable": {
                    k: {
                        "variable": v.variable,
                        "section": v.section,
                        "policyengine": v.policyengine,
                        "taxsim": v.taxsim,
                        "taxcalc": v.taxcalc,
                        "prd": v.prd,
                        "consensus": v.consensus,
                    }
                    for k, v in self.alignment.by_variable.items()
                },
                "by_validator": self.alignment.by_validator,
            },
            "coverage": {
                "implemented": self.coverage.implemented,
                "total": self.coverage.total,
                "percentage": self.coverage.percentage,
                "by_section": self.coverage.by_section,
            },
            "quality": {
                "test_coverage": self.quality.test_coverage,
                "no_literals_pass": self.quality.no_literals_pass,
                "all_imports_valid": self.quality.all_imports_valid,
                "all_dtypes_valid": self.quality.all_dtypes_valid,
                "overall_score": self.quality.overall_score,
                "issues": [
                    {
                        "file": i.file,
                        "line": i.line,
                        "category": i.category,
                        "severity": i.severity,
                        "message": i.message,
                    }
                    for i in self.quality.issues
                ],
            },
            "review": (
                {
                    "overall_score": self.review.overall_score,
                    "accuracy": self.review.accuracy,
                    "completeness": self.review.completeness,
                    "parameterization": self.review.parameterization,
                    "test_quality": self.review.test_quality,
                    "feedback": self.review.feedback,
                    "reviewed_files": self.review.reviewed_files,
                }
                if self.review
                else None
            ),
        }


@dataclass
class Checkpoint:
    """Saved validation checkpoint."""

    timestamp: str
    git_commit: str
    scores: dict[str, float]  # alignment, coverage, quality, review
    details: dict  # Full HarnessResult as dict

    @classmethod
    def from_result(cls, result: HarnessResult) -> "Checkpoint":
        """Create checkpoint from harness result."""
        return cls(
            timestamp=result.timestamp,
            git_commit=result.git_commit,
            scores={
                "alignment": result.alignment.overall_rate,
                "coverage": result.coverage.percentage,
                "quality": result.quality.overall_score / 100,
                "review": result.review.overall_score / 10 if result.review else 0.0,
            },
            details=result.to_dict(),
        )


@dataclass
class Delta:
    """Difference between two checkpoints."""

    before: Checkpoint
    after: Checkpoint

    @property
    def alignment_delta(self) -> float:
        return self.after.scores.get("alignment", 0) - self.before.scores.get("alignment", 0)

    @property
    def coverage_delta(self) -> float:
        return self.after.scores.get("coverage", 0) - self.before.scores.get("coverage", 0)

    @property
    def quality_delta(self) -> float:
        return self.after.scores.get("quality", 0) - self.before.scores.get("quality", 0)

    @property
    def review_delta(self) -> float:
        return self.after.scores.get("review", 0) - self.before.scores.get("review", 0)

    def has_regression(self, threshold: float = 0.01) -> bool:
        """Check if any metric regressed beyond threshold."""
        return (
            self.alignment_delta < -threshold
            or self.coverage_delta < -threshold
            or self.quality_delta < -threshold
        )
