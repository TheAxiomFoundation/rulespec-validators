# Scope of rac-validators

This document clarifies what `rac-validators` is responsible for and what
belongs elsewhere in the Rules Foundation toolchain. It exists because an
internal review flagged that parts of the `harness/quality/` tree overlap
with work that `rac-compile` already does (or could do) during compilation.

## What rac-validators IS

`rac-validators` is an **oracle-consensus** framework for encoded statutes.
Given a RAC encoding of a tax or benefit rule and a set of test cases with
expected values, it:

1. **Runs the encoding against multiple independent oracle systems**
   (PolicyEngine, TAXSIM, PSL / Tax-Calculator, Yale Tax-Simulator).
2. **Aggregates the results** into a single consensus value, confidence
   score, and a reward signal in the range `[-1.0, +1.0]` suitable for
   training AI encoders.
3. **Flags upstream bugs** when the Claude encoder is highly confident,
   the citation is clear, and one or more oracle systems disagree with
   the expected value.

In short: `rac-validators` answers "*does this encoding behave the same
way as every established calculator of record, and if not, which one is
wrong?*" It operates at **runtime**, on **calculated outputs**.

## What rac-validators is NOT

`rac-validators` is **not** a compile-time or static-analysis tool. It
does not (and should not) own:

- Grammar/syntax validation of `.rac` files — that is `rac-syntax`.
- Type checking, import resolution, schema checks, or any other
  compile-time linting of an encoding — that is `rac-compile`.
- The RAC DSL semantics themselves — that is `rac`.

If a check can be performed by reading the source of a `.rac` file
without running any oracle, it probably belongs in `rac-compile`.

## Known overlap

The following modules currently duplicate logic that belongs in (or is
already in) `rac-compile`:

| Module | Overlap with rac-compile |
|---|---|
| `harness/quality/imports.py` | Validates that `path#variable` imports resolve. This is a static-source check. |
| `harness/quality/schema.py` | Checks YAML/schema shape of `.rac` files. Compile-time concern. |
| `harness/quality/grounding.py` | Citation / source-grounding checks. Partially static; runtime grounding stays here. |

Each of these modules has been tagged with a module-level comment:

```python
# NOTE: This may move to rac-compile. See docs/scope.md.
```

**Migration is under consideration but is not scheduled.** The overlap is
tolerated for now because:

- The harness needs *some* quality gate before running oracles — a
  malformed `.rac` file should fail fast rather than cause opaque
  oracle errors.
- Moving checks requires a coordinated release of `rac-compile` and
  a new integration surface here. That work is tracked separately.

If you add new quality checks, prefer extending `rac-compile` and
importing from it rather than adding logic under `harness/quality/`.

## Decision rule

When deciding where a new check belongs:

- **Reads only the `.rac` source?** → `rac-compile`.
- **Runs an oracle or compares numerical outputs?** → `rac-validators`.
- **Concerns the DSL grammar itself?** → `rac-syntax` or `rac`.
