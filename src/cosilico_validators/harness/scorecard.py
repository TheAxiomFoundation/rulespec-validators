"""Scorecard generation for PR comments."""

from typing import Optional

from . import Checkpoint, Delta, HarnessResult


def format_delta(value: float, is_percentage: bool = True) -> str:
    """Format a delta value with arrow indicator."""
    if abs(value) < 0.001:
        return "-"

    formatted = f"{value * 100:+.1f}%" if is_percentage else f"{value:+.2f}"

    if value > 0:
        return f"{formatted} :arrow_up:"
    else:
        return f"{formatted} :arrow_down:"


def format_percentage(value: float) -> str:
    """Format a percentage value."""
    return f"{value * 100:.1f}%"


def generate_scorecard(
    result: HarnessResult,
    baseline: Optional[Checkpoint] = None,
) -> str:
    """Generate markdown scorecard.

    Args:
        result: Current harness result
        baseline: Previous checkpoint to compare against (optional)

    Returns:
        Markdown formatted scorecard
    """
    lines = []
    lines.append("## Cosilico Validation Scorecard")
    lines.append("")

    # Summary table
    lines.append("### Summary")
    lines.append("")
    lines.append("| Metric | Current | Delta |")
    lines.append("|--------|---------|-------|")

    if baseline:
        delta = Delta(before=baseline, after=Checkpoint.from_result(result))
        lines.append(
            f"| Alignment | {format_percentage(result.alignment.overall_rate)} | "
            f"{format_delta(delta.alignment_delta)} |"
        )
        lines.append(
            f"| Coverage | {result.coverage.implemented}/{result.coverage.total} | "
            f"{format_delta(delta.coverage_delta)} |"
        )
        lines.append(
            f"| Quality | {result.quality.overall_score:.0f}% | "
            f"{format_delta(delta.quality_delta)} |"
        )
        if result.review:
            lines.append(
                f"| Review | {result.review.overall_score:.1f}/10 | "
                f"{format_delta(delta.review_delta, is_percentage=False)} |"
            )
    else:
        lines.append(f"| Alignment | {format_percentage(result.alignment.overall_rate)} | - |")
        lines.append(
            f"| Coverage | {result.coverage.implemented}/{result.coverage.total} "
            f"({format_percentage(result.coverage.percentage)}) | - |"
        )
        lines.append(f"| Quality | {result.quality.overall_score:.0f}% | - |")
        if result.review:
            lines.append(f"| Review | {result.review.overall_score:.1f}/10 | - |")

    lines.append("")

    # Alignment by variable
    if result.alignment.by_variable:
        lines.append("### Alignment by Variable")
        lines.append("")
        lines.append("| Variable | Section | PolicyEngine | TAXSIM | Consensus |")
        lines.append("|----------|---------|--------------|--------|-----------|")

        for var_name, var_align in sorted(result.alignment.by_variable.items()):
            pe = format_percentage(var_align.policyengine) if var_align.policyengine else "-"
            ts = format_percentage(var_align.taxsim) if var_align.taxsim else "-"
            cons = format_percentage(var_align.consensus)
            lines.append(f"| {var_name} | {var_align.section} | {pe} | {ts} | {cons} |")

        lines.append("")

    # Coverage by section
    if result.coverage.by_section:
        lines.append("### Coverage by Section")
        lines.append("")
        lines.append("| Section | Implemented | Total | Rate |")
        lines.append("|---------|-------------|-------|------|")

        for section, (impl, total) in sorted(result.coverage.by_section.items()):
            rate = impl / total if total > 0 else 0
            lines.append(f"| {section} | {impl} | {total} | {format_percentage(rate)} |")

        lines.append("")

    # Quality metrics
    lines.append("### Quality Metrics")
    lines.append("")
    lines.append(f"- Test Coverage: {format_percentage(result.quality.test_coverage)}")
    lines.append(f"- No Hardcoded Literals: {'PASS :white_check_mark:' if result.quality.no_literals_pass else 'FAIL :x:'}")
    lines.append(f"- All Imports Valid: {'PASS :white_check_mark:' if result.quality.all_imports_valid else 'FAIL :x:'}")
    lines.append(f"- All Dtypes Valid: {'PASS :white_check_mark:' if result.quality.all_dtypes_valid else 'FAIL :x:'}")
    lines.append("")

    # Quality issues
    if result.quality.issues:
        lines.append("#### Issues Found")
        lines.append("")
        for issue in result.quality.issues[:10]:  # Limit to 10
            icon = ":x:" if issue.severity == "error" else ":warning:"
            loc = f":{issue.line}" if issue.line else ""
            lines.append(f"- {icon} `{issue.file}{loc}`: {issue.message}")
        if len(result.quality.issues) > 10:
            lines.append(f"- ... and {len(result.quality.issues) - 10} more issues")
        lines.append("")

    # Agent review
    if result.review:
        lines.append("### Agent Review")
        lines.append("")
        lines.append(f"> **Score: {result.review.overall_score:.1f}/10**")
        lines.append(">")
        for line in result.review.feedback.split("\n"):
            lines.append(f"> {line}")
        lines.append("")

        if result.review.reviewed_files:
            lines.append("**Reviewed files:**")
            for f in result.review.reviewed_files:
                lines.append(f"- `{f}`")
            lines.append("")

    # Footer
    lines.append("---")
    lines.append(f"*Generated at {result.timestamp} from commit `{result.git_commit}`*")

    return "\n".join(lines)


def generate_compact_scorecard(result: HarnessResult, baseline: Optional[Checkpoint] = None) -> str:
    """Generate a compact one-line summary for logs."""
    parts = [
        f"Alignment: {result.alignment.overall_rate * 100:.1f}%",
        f"Coverage: {result.coverage.implemented}/{result.coverage.total}",
        f"Quality: {result.quality.overall_score:.0f}%",
    ]

    if result.review:
        parts.append(f"Review: {result.review.overall_score:.1f}/10")

    if baseline:
        delta = Delta(before=baseline, after=Checkpoint.from_result(result))
        if delta.alignment_delta != 0:
            parts[0] += f" ({delta.alignment_delta * 100:+.1f}%)"

    return " | ".join(parts)
