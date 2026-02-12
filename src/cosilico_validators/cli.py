"""CLI for cosilico-validators."""

import json
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from cosilico_validators.consensus.engine import ConsensusEngine, ConsensusLevel
from cosilico_validators.validators.base import TestCase

console = Console()


def load_validators(include_policyengine: bool = True, include_taxsim: bool = True):
    """Load available validators."""
    validators = []

    if include_taxsim:
        from cosilico_validators.validators.taxsim import TaxsimValidator
        validators.append(TaxsimValidator())

    if include_policyengine:
        try:
            from cosilico_validators.validators.policyengine import PolicyEngineValidator
            validators.append(PolicyEngineValidator())
        except ImportError:
            console.print("[yellow]PolicyEngine not installed, skipping[/yellow]")

    return validators


@click.group()
def cli():
    """Multi-system tax/benefit validation for Cosilico DSL encodings."""
    pass


@cli.command()
@click.argument("test_file", type=click.Path(exists=True))
@click.option("--variable", "-v", required=True, help="Variable to validate (e.g., eitc, ctc)")
@click.option("--year", "-y", default=2024, help="Tax year")
@click.option("--tolerance", "-t", default=15.0, help="Dollar tolerance for matching")
@click.option("--no-policyengine", is_flag=True, help="Skip PolicyEngine validator")
@click.option("--no-taxsim", is_flag=True, help="Skip TAXSIM validator")
@click.option("--claude-confidence", type=float, help="Claude's confidence in expected value (0-1)")
@click.option("--output", "-o", type=click.Path(), help="Output file for results (JSON)")
def validate(test_file, variable, year, tolerance, no_policyengine, no_taxsim, claude_confidence, output):
    """Validate test cases against multiple systems."""
    # Load test cases
    test_path = Path(test_file)
    if test_path.suffix == ".json":
        with open(test_path) as f:
            test_data = json.load(f)
    elif test_path.suffix in [".yaml", ".yml"]:
        import yaml
        with open(test_path) as f:
            test_data = yaml.safe_load(f)
    else:
        raise click.ClickException(f"Unsupported file format: {test_path.suffix}")

    # Convert to TestCase objects
    test_cases = []
    if isinstance(test_data, list):
        for tc in test_data:
            test_cases.append(TestCase(
                name=tc.get("name", "unnamed"),
                inputs=tc.get("inputs", {}),
                expected=tc.get("expected", {}),
                citation=tc.get("citation"),
                notes=tc.get("notes"),
            ))
    elif isinstance(test_data, dict) and "test_cases" in test_data:
        for tc in test_data["test_cases"]:
            test_cases.append(TestCase(
                name=tc.get("name", "unnamed"),
                inputs=tc.get("inputs", {}),
                expected=tc.get("expected", {}),
                citation=tc.get("citation"),
                notes=tc.get("notes"),
            ))

    if not test_cases:
        raise click.ClickException("No test cases found in file")

    # Load validators
    validators = load_validators(
        include_policyengine=not no_policyengine,
        include_taxsim=not no_taxsim,
    )

    if not validators:
        raise click.ClickException("No validators available")

    # Create consensus engine
    engine = ConsensusEngine(validators, tolerance=tolerance)

    # Run validation
    results = []
    for tc in test_cases:
        result = engine.validate(tc, variable, year, claude_confidence)
        results.append(result)

    # Display results
    display_results(results)

    # Save output if requested
    if output:
        output_data = []
        for r in results:
            output_data.append({
                "test_case": r.test_case.name,
                "variable": r.variable,
                "expected": r.expected_value,
                "consensus_value": r.consensus_value,
                "consensus_level": r.consensus_level.value,
                "reward_signal": r.reward_signal,
                "confidence": r.confidence,
                "matches_expected": r.matches_expected,
                "validator_results": {
                    name: {
                        "calculated": vr.calculated_value,
                        "error": vr.error,
                        "success": vr.success,
                    }
                    for name, vr in r.validator_results.items()
                },
                "potential_bugs": r.potential_bugs,
            })

        with open(output, "w") as f:
            json.dump(output_data, f, indent=2)
        console.print(f"\n[green]Results saved to {output}[/green]")

    # Summary statistics
    display_summary(results)


def display_results(results):
    """Display validation results in a table."""
    table = Table(title="Validation Results")
    table.add_column("Test Case", style="cyan")
    table.add_column("Expected", justify="right")
    table.add_column("Consensus", justify="right")
    table.add_column("Level", style="magenta")
    table.add_column("Reward", justify="right")
    table.add_column("Match", justify="center")

    level_colors = {
        ConsensusLevel.FULL_AGREEMENT: "green",
        ConsensusLevel.PRIMARY_CONFIRMED: "green",
        ConsensusLevel.MAJORITY_AGREEMENT: "yellow",
        ConsensusLevel.DISAGREEMENT: "red",
        ConsensusLevel.POTENTIAL_UPSTREAM_BUG: "blue",
    }

    for r in results:
        consensus_str = f"${r.consensus_value:,.0f}" if r.consensus_value else "N/A"
        level_color = level_colors.get(r.consensus_level, "white")
        match_str = "✓" if r.matches_expected else "✗"
        match_color = "green" if r.matches_expected else "red"

        table.add_row(
            r.test_case.name[:30],
            f"${r.expected_value:,.0f}",
            consensus_str,
            f"[{level_color}]{r.consensus_level.value}[/{level_color}]",
            f"{r.reward_signal:+.2f}",
            f"[{match_color}]{match_str}[/{match_color}]",
        )

    console.print(table)

    # Show potential bugs
    all_bugs = []
    for r in results:
        all_bugs.extend(r.potential_bugs)

    if all_bugs:
        console.print("\n")
        bug_panel = Panel(
            "\n".join([
                f"• {bug['validator']}: expected ${bug['expected']:,.0f}, got ${bug['actual']:,.0f} "
                f"(diff: ${bug['difference']:,.0f})"
                for bug in all_bugs
            ]),
            title="[bold red]Potential Upstream Bugs Detected[/bold red]",
            border_style="red",
        )
        console.print(bug_panel)


def display_summary(results):
    """Display summary statistics."""
    total = len(results)
    matches = sum(1 for r in results if r.matches_expected)
    avg_reward = sum(r.reward_signal for r in results) / total if total else 0
    avg_confidence = sum(r.confidence for r in results) / total if total else 0

    level_counts = {}
    for r in results:
        level_counts[r.consensus_level.value] = level_counts.get(r.consensus_level.value, 0) + 1

    console.print("\n")
    summary = f"""[bold]Summary[/bold]
Total tests: {total}
Matches: {matches}/{total} ({matches/total*100:.1f}%)
Average reward: {avg_reward:+.3f}
Average confidence: {avg_confidence:.1%}

Consensus levels:
"""
    for level, count in sorted(level_counts.items()):
        summary += f"  {level}: {count}\n"

    console.print(Panel(summary, border_style="blue"))


@cli.command()
@click.option("--variable", "-v", help="Variable to check")
def validators(variable):
    """List available validators and their supported variables."""
    validators = load_validators()

    table = Table(title="Available Validators")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Variables")

    for v in validators:
        vars_list = sorted(v.supported_variables) if hasattr(v, "supported_variables") else ["(dynamic)"]
        if variable:
            supports = v.supports_variable(variable)
            vars_str = f"[green]✓ Supports {variable}[/green]" if supports else f"[red]✗ No {variable}[/red]"
        else:
            vars_str = ", ".join(vars_list[:5])
            if len(vars_list) > 5:
                vars_str += f" (+{len(vars_list)-5} more)"

        table.add_row(v.name, v.validator_type.value, vars_str)

    console.print(table)


@cli.command()
@click.argument("results_file", type=click.Path(exists=True))
@click.option("--repo", "-r", help="Target repo for issues (e.g., PolicyEngine/policyengine-us)")
@click.option("--dry-run", is_flag=True, help="Show what would be filed without creating issues")
def file_issues(results_file, repo, dry_run):
    """File GitHub issues for potential upstream bugs."""
    with open(results_file) as f:
        results = json.load(f)

    bugs = []
    for r in results:
        bugs.extend(r.get("potential_bugs", []))

    if not bugs:
        console.print("[green]No potential bugs to file![/green]")
        return

    console.print(f"[bold]Found {len(bugs)} potential bugs[/bold]\n")

    for bug in bugs:
        title = f"Potential calculation error in {bug['test_case']}"
        body = f"""## Bug Report (Auto-generated)

**Test Case:** {bug['test_case']}
**Variable:** Calculated value mismatch

### Expected vs Actual
- **Expected (from statute):** ${bug['expected']:,.2f}
- **Calculated:** ${bug['actual']:,.2f}
- **Difference:** ${bug['difference']:,.2f}

### Citation
{bug.get('citation', 'N/A')}

### Test Inputs
```json
{json.dumps(bug.get('inputs', {}), indent=2)}
```

### Confidence
Claude encoding confidence: {bug.get('claude_confidence', 'N/A')}

---
*This issue was automatically generated by cosilico-validators based on multi-system consensus analysis.*
"""
        console.print(Panel(
            f"[bold]{title}[/bold]\n\n{body[:500]}...",
            title=f"Issue for {bug['validator']}",
            border_style="yellow" if dry_run else "green",
        ))

        if not dry_run and repo:
            # TODO: Actually file the issue using GitHub API
            console.print(f"[yellow]Would file to {repo} (not implemented yet)[/yellow]")

    if dry_run:
        console.print("\n[yellow]Dry run - no issues were filed[/yellow]")


@cli.command()
@click.option("--year", "-y", default=2024, help="Tax year")
@click.option("--output", "-o", type=click.Path(), help="Output JSON file for dashboard")
def compare_aligned(year, output):
    """Compare using common dataset (isolates rule differences).

    Uses PolicyEngine's input data for both systems, ensuring identical
    inputs so differences reflect only rule implementation.
    """
    from cosilico_validators.comparison import run_aligned_comparison

    console.print("\n[bold]Aligned Comparison (Common Dataset)[/bold]")
    console.print(f"Year: {year}")
    console.print("Using PE inputs for both systems to isolate rule differences\n")

    try:
        dashboard = run_aligned_comparison(year=year)
    except ImportError as e:
        raise click.ClickException(str(e)) from e

    # Display summary
    table = Table(title="Rules Alignment Results")
    table.add_column("Variable", style="cyan")
    table.add_column("Match Rate", justify="right")
    table.add_column("MAE", justify="right")
    table.add_column("Cosilico", justify="right")
    table.add_column("PE", justify="right")
    table.add_column("Diff", justify="right")

    for var_result in dashboard["variables"]:
        match_pct = var_result["match_rate"] * 100
        match_color = "green" if match_pct > 90 else "yellow" if match_pct > 75 else "red"
        diff_b = var_result["difference_billions"]
        diff_color = "green" if abs(diff_b) < 10 else "yellow" if abs(diff_b) < 50 else "red"

        table.add_row(
            var_result["variable"],
            f"[{match_color}]{match_pct:.1f}%[/{match_color}]",
            f"${var_result['mean_absolute_error']:,.0f}",
            f"${var_result['cosilico_weighted_total']/1e9:.1f}B",
            f"${var_result['policyengine_weighted_total']/1e9:.1f}B",
            f"[{diff_color}]{diff_b:+.1f}B[/{diff_color}]",
        )

    console.print(table)

    summary = dashboard["summary"]
    console.print(f"\n[bold]Overall:[/bold] {summary['overall_match_rate']*100:.1f}% match rate")
    console.print(f"Records: {summary['total_records']:,}")

    if output:
        with open(output, "w") as f:
            json.dump(dashboard, f, indent=2)
        console.print(f"\n[green]Results saved to {output}[/green]")


@cli.command()
@click.option("--year", "-y", default=2024, help="Tax year")
@click.option("--tolerance", "-t", default=1.0, help="Dollar tolerance for matching")
@click.option("--variables", "-v", multiple=True, help="Variables to compare (default: eitc, income_tax, agi)")
@click.option("--output", "-o", type=click.Path(), help="Output JSON file for dashboard")
def compare(year, tolerance, variables, output):
    """Compare Cosilico vs PolicyEngine record-by-record on CPS data."""
    from cosilico_validators.comparison import run_full_comparison

    variables_list = list(variables) if variables else None

    console.print("\n[bold]Cosilico vs PolicyEngine Record Comparison[/bold]")
    console.print(f"Year: {year}, Tolerance: ${tolerance:.2f}\n")

    try:
        dashboard = run_full_comparison(
            variables=variables_list,
            year=year,
            tolerance=tolerance,
        )
    except ImportError as e:
        raise click.ClickException(str(e)) from e

    # Display summary
    table = Table(title="Comparison Results")
    table.add_column("Variable", style="cyan")
    table.add_column("Match Rate", justify="right")
    table.add_column("MAE", justify="right")
    table.add_column("Records", justify="right")

    for var_result in dashboard["variables"]:
        if "error" in var_result:
            table.add_row(
                var_result["variable"],
                "[red]ERROR[/red]",
                "-",
                "-",
            )
        else:
            match_pct = var_result.get("match_rate", 0) * 100
            match_color = "green" if match_pct > 90 else "yellow" if match_pct > 75 else "red"
            table.add_row(
                var_result["variable"],
                f"[{match_color}]{match_pct:.1f}%[/{match_color}]",
                f"${var_result.get('mean_absolute_error', 0):,.0f}",
                f"{var_result.get('n_records', 0):,}",
            )

    console.print(table)

    # Summary stats
    summary = dashboard.get("summary", {})
    console.print(f"\n[bold]Overall:[/bold] {summary.get('overall_match_rate', 0)*100:.1f}% match rate")
    console.print(f"Total records: {summary.get('total_records', 0):,}")

    # Save output
    if output:
        with open(output, "w") as f:
            json.dump(dashboard, f, indent=2)
        console.print(f"\n[green]Dashboard saved to {output}[/green]")


@cli.command()
@click.option("--year", "-y", default=2024, help="Tax year")
@click.option("--output", "-o", type=click.Path(), help="Output JSON file for dashboard")
def dashboard(year, output):
    """Export validation results to cosilico.ai dashboard format.

    Generates a ValidationResults JSON file compatible with the
    cosilico.ai/validation dashboard page.

    Example:
        cosilico-validators dashboard -o validation-results.json
        cp validation-results.json ~/CosilicoAI/cosilico.ai/public/data/
    """
    from pathlib import Path

    from cosilico_validators.dashboard_export import run_export

    console.print("\n[bold]Dashboard Export[/bold]")
    console.print(f"Year: {year}\n")

    output_path = Path(output) if output else None

    try:
        data = run_export(year=year, output_path=output_path)
    except ImportError as e:
        raise click.ClickException(str(e)) from e

    # Display summary
    table = Table(title="Validation Summary")
    table.add_column("Variable", style="cyan")
    table.add_column("Section")
    table.add_column("Match Rate", justify="right")
    table.add_column("MAE", justify="right")

    for section in data["sections"]:
        match_pct = section["summary"]["matchRate"] * 100
        match_color = "green" if match_pct > 90 else "yellow" if match_pct > 75 else "red"
        table.add_row(
            section["variable"],
            section["section"],
            f"[{match_color}]{match_pct:.1f}%[/{match_color}]",
            f"${section['summary']['meanAbsoluteError']:,.0f}",
        )

    console.print(table)

    overall = data["overall"]
    console.print(f"\n[bold]Overall:[/bold] {overall['matchRate']*100:.1f}% match rate")
    console.print(f"Households: {overall['totalHouseholds']:,}")
    console.print(f"Mean Absolute Error: ${overall['meanAbsoluteError']:,.2f}")

    if output_path:
        console.print(f"\n[green]Dashboard JSON saved to {output_path}[/green]")


# ============================================================================
# HARNESS COMMANDS
# ============================================================================


@cli.group()
def harness():
    """Validation harness for encoding work."""
    pass


@harness.command("run")
@click.option("--only", type=click.Choice(["alignment", "quality", "review"]), help="Run only specific check")
@click.option("--output", "-o", type=click.Path(), help="Save results to JSON file")
@click.option("--baseline", "-b", type=click.Path(), help="Compare against baseline")
def harness_run(only, output, baseline):
    """Run validation harness.

    Example:
        cosilico-validators harness run
        cosilico-validators harness run --only quality
        cosilico-validators harness run -o current.json -b baselines/main.json
    """
    from cosilico_validators.harness.checkpoint import (
        load_checkpoint,
        save_checkpoint,
    )
    from cosilico_validators.harness.runner import run_harness
    from cosilico_validators.harness.scorecard import generate_compact_scorecard

    console.print("\n[bold]Cosilico Validation Harness[/bold]")
    if only:
        console.print(f"Running: {only}")
    console.print()

    # Run harness
    result = run_harness(only=only)

    # Load baseline if provided
    baseline_checkpoint = None
    if baseline:
        baseline_path = Path(baseline)
        baseline_checkpoint = load_checkpoint(baseline_path)
        if baseline_checkpoint:
            console.print(f"[dim]Baseline: {baseline_checkpoint.git_commit}[/dim]")

    # Display compact summary
    summary = generate_compact_scorecard(result, baseline_checkpoint)
    console.print(f"\n[bold]{summary}[/bold]\n")

    # Display alignment table
    if result.alignment.by_variable:
        table = Table(title="Alignment by Variable")
        table.add_column("Variable", style="cyan")
        table.add_column("Section")
        table.add_column("PolicyEngine", justify="right")
        table.add_column("Consensus", justify="right")

        for var_name, var_align in sorted(result.alignment.by_variable.items()):
            pe = f"{var_align.policyengine * 100:.1f}%" if var_align.policyengine else "-"
            cons = f"{var_align.consensus * 100:.1f}%"
            table.add_row(var_name, var_align.section, pe, cons)

        console.print(table)

    # Display quality issues
    if result.quality.issues:
        console.print(f"\n[yellow]Quality Issues: {len(result.quality.issues)}[/yellow]")
        for issue in result.quality.issues[:5]:
            icon = "✗" if issue.severity == "error" else "⚠"
            console.print(f"  {icon} {issue.file}:{issue.line}: {issue.message}")
        if len(result.quality.issues) > 5:
            console.print(f"  ... and {len(result.quality.issues) - 5} more")

    # Save output if requested
    if output:
        output_path = Path(output)
        save_checkpoint(result, output_path)
        console.print(f"\n[green]Results saved to {output}[/green]")


@harness.command("checkpoint")
@click.option("--save", "-s", type=click.Path(), help="Save current state as checkpoint")
@click.option("--name", "-n", default="latest", help="Checkpoint name (default: latest)")
def harness_checkpoint(save, name):
    """Save or manage checkpoints.

    Example:
        cosilico-validators harness checkpoint --save baselines/main.json
        cosilico-validators harness checkpoint --name main
    """
    from cosilico_validators.harness.checkpoint import save_baseline
    from cosilico_validators.harness.runner import run_harness

    if save:
        console.print(f"[bold]Saving checkpoint to {save}[/bold]")
        result = run_harness()
        from cosilico_validators.harness.checkpoint import save_checkpoint

        save_checkpoint(result, Path(save))
        console.print("[green]Saved![/green]")
    else:
        # Save to named baseline
        console.print(f"[bold]Saving checkpoint: {name}[/bold]")
        result = run_harness()
        path = save_baseline(result, name)
        console.print(f"[green]Saved to {path}[/green]")


@harness.command("compare")
@click.option("--baseline", "-b", type=click.Path(exists=True), required=True, help="Baseline checkpoint")
@click.option("--current", "-c", type=click.Path(), help="Current checkpoint (default: run now)")
def harness_compare(baseline, current):
    """Compare current state against baseline.

    Example:
        cosilico-validators harness compare -b baselines/main.json
    """
    from cosilico_validators.harness.checkpoint import (
        Checkpoint,
        compare_checkpoints,
        load_checkpoint,
    )
    from cosilico_validators.harness.runner import run_harness

    console.print("\n[bold]Comparing against baseline[/bold]")

    # Load baseline
    baseline_cp = load_checkpoint(Path(baseline))
    if not baseline_cp:
        raise click.ClickException(f"Could not load baseline: {baseline}")

    console.print(f"Baseline: {baseline_cp.git_commit} ({baseline_cp.timestamp[:10]})")

    # Get current
    if current:
        current_cp = load_checkpoint(Path(current))
        if not current_cp:
            raise click.ClickException(f"Could not load current: {current}")
    else:
        console.print("Running validation...")
        result = run_harness()
        current_cp = Checkpoint.from_result(result)

    console.print(f"Current: {current_cp.git_commit}\n")

    # Compare
    delta = compare_checkpoints(baseline_cp, current_cp)

    # Display comparison table
    table = Table(title="Comparison")
    table.add_column("Metric", style="cyan")
    table.add_column("Before", justify="right")
    table.add_column("After", justify="right")
    table.add_column("Delta", justify="right")

    def format_delta(d):
        if abs(d) < 0.001:
            return "-"
        color = "green" if d > 0 else "red"
        return f"[{color}]{d * 100:+.1f}%[/{color}]"

    table.add_row(
        "Alignment",
        f"{baseline_cp.scores.get('alignment', 0) * 100:.1f}%",
        f"{current_cp.scores.get('alignment', 0) * 100:.1f}%",
        format_delta(delta.alignment_delta),
    )
    table.add_row(
        "Coverage",
        f"{baseline_cp.scores.get('coverage', 0) * 100:.1f}%",
        f"{current_cp.scores.get('coverage', 0) * 100:.1f}%",
        format_delta(delta.coverage_delta),
    )
    table.add_row(
        "Quality",
        f"{baseline_cp.scores.get('quality', 0) * 100:.1f}%",
        f"{current_cp.scores.get('quality', 0) * 100:.1f}%",
        format_delta(delta.quality_delta),
    )

    console.print(table)

    if delta.has_regression():
        console.print("\n[yellow]Warning: Regression detected[/yellow]")
    else:
        console.print("\n[green]No regressions[/green]")


@harness.command("scorecard")
@click.option("--before", "-b", type=click.Path(exists=True), help="Before checkpoint")
@click.option("--after", "-a", type=click.Path(exists=True), help="After checkpoint (or run now)")
@click.option("--output", "-o", type=click.Path(), help="Output markdown file")
def harness_scorecard(before, after, output):
    """Generate PR scorecard.

    Example:
        cosilico-validators harness scorecard -b main.json -a current.json -o scorecard.md
    """
    from cosilico_validators.harness.checkpoint import load_checkpoint
    from cosilico_validators.harness.runner import run_harness
    from cosilico_validators.harness.scorecard import generate_scorecard

    console.print("\n[bold]Generating Scorecard[/bold]\n")

    # Load before checkpoint
    baseline = None
    if before:
        baseline = load_checkpoint(Path(before))

    # Get after result
    if after:
        after_cp = load_checkpoint(Path(after))
        if not after_cp:
            raise click.ClickException(f"Could not load: {after}")
        # Reconstruct result from checkpoint details
        # For now, just run fresh
        result = run_harness()
    else:
        result = run_harness()

    # Generate scorecard
    scorecard = generate_scorecard(result, baseline)

    if output:
        with open(output, "w") as f:
            f.write(scorecard)
        console.print(f"[green]Scorecard saved to {output}[/green]")
    else:
        console.print(scorecard)


if __name__ == "__main__":  # pragma: no cover
    cli()
