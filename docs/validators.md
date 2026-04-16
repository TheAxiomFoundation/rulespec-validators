# Oracle validators

`rac-validators` runs an encoded rule through multiple independent tax
calculators (the "oracles") and compares their outputs. Each oracle is
a separate validator class under
`src/cosilico_validators/validators/`. This document catalogs them so
that adding new oracles, debugging failing ones, and triaging CI
problems is straightforward.

## Validator summary

| Validator | Class | Type | Data source | Install footprint | Network? |
|---|---|---|---|---|---|
| PolicyEngine | `PolicyEngineValidator` | REFERENCE | `policyengine-us` (Python pkg, open source) | Large: pulls `policyengine-us` + `policyengine-core` | No (runs locally once installed) |
| TAXSIM | `TaxsimValidator` | REFERENCE | NBER TAXSIM 35 — federal & state income tax | None (web API) OR a downloaded binary | Yes by default (web API). Optional local executable. |
| PSL Tax-Calculator | `TaxcalcValidator` | SUPPLEMENTARY | `taxcalc` (Policy Simulation Library) | Medium: `taxcalc` + `behresp` | No once installed |
| Yale Budget Lab | `YaleValidator` | SUPPLEMENTARY | R-based Tax-Simulator repo | Large & out-of-band: requires R 4.0+, `tidyverse`, `yaml`, `data.table`, and a local clone of [Budget-Lab-Yale/Tax-Simulator](https://github.com/Budget-Lab-Yale/Tax-Simulator) | No once installed |
| TaxAct | *(conceptual)* | PRIMARY | Commercial consumer tax software | — | — |

### TaxAct note

`ValidatorType.PRIMARY` is documented in `validators/base.py` as
"TaxAct - ground truth", but **there is no `TaxActValidator` class
today**. TaxAct is the aspirational primary oracle; until a validator
lands, PolicyEngine is the closest stand-in and the consensus engine's
``primary_weight`` multiplier applies to whichever validator is
registered as `PRIMARY`.

## Per-validator notes

### PolicyEngine

- **Install**: `pip install "cosilico-validators[policyengine]"` or
  `uv pip install policyengine-us policyengine-core`.
- **Coverage**: federal & state income tax, EITC, CTC, SNAP, Medicaid,
  TANF, AGI, standard deduction variants. See `VARIABLE_MAPPING` in
  `validators/policyengine.py` for the exact mapping from canonical
  variable names to PolicyEngine variables.
- **Fragility points**:
  - `policyengine-us` pins tend to drift; when CI breaks mysteriously,
    check whether a new `policyengine-us` release changed a default
    parameter value.
  - State coverage varies by state and year.
  - Household inputs must be shaped as PolicyEngine entities
    (Person / TaxUnit / Household / Family) — schema mismatches
    produce silent zeros rather than errors.

### TAXSIM

- **Install**: no Python deps. Default mode hits the NBER web API at
  `taxsim.nber.org/taxsim35/`.
- **Coverage**: federal income tax, state income tax, FICA; limited
  benefits support.
- **Fragility points**:
  - **Network dependency** — tests that hit TAXSIM must be marked
    `@pytest.mark.integration`. Flaky CI is almost always NBER rate
    limiting or transient 5xx.
  - Variable IDs are numeric (TAXSIM output columns). Updates to
    TAXSIM's output schema silently change what each column means.
  - Optional local-binary mode requires downloading a per-platform
    executable from [NBER](https://taxsim.nber.org/taxsim35/); set
    `TAXSIM_EXECUTABLE=/path/to/bin` to use it.

### PSL Tax-Calculator

- **Install**: `pip install "cosilico-validators[psl]"` pulls
  `taxcalc` and `behresp`.
- **Coverage**: U.S. federal income and payroll taxes. No state taxes.
  No non-tax benefits.
- **Fragility points**:
  - `behresp` is a hard runtime dependency of `taxcalc` even when no
    behavioral response is needed — missing `behresp` yields a cryptic
    import error.
  - CPS-based microdata assumptions differ from PolicyEngine's; small
    discrepancies on edge cases are expected and should not be chased
    as bugs until differences exceed the engine's tolerance.

### Yale Budget Lab Tax-Simulator

- **Install**: out-of-band. Requires R 4.0+, the R packages listed
  above, and a local clone of Budget-Lab-Yale/Tax-Simulator. Point the
  validator at the clone via its constructor.
- **Coverage**: U.S. federal income tax, behavioral responses, tax
  reform scenarios.
- **Fragility points**:
  - Subprocess boundary — validator shells out to `Rscript`. `R` not
    being on `PATH` is the most common failure.
  - I/O is JSON over tempfiles; malformed household inputs produce R
    tracebacks that surface as non-zero exit codes.
  - Upstream repo is under active research; breaking changes are
    occasional. Pin to a known-good commit.

## Adding a new oracle

1. Subclass `BaseValidator` in a new file under `validators/`.
2. Declare a `ValidatorType` — use `SUPPLEMENTARY` unless there is a
   strong reason otherwise.
3. Implement `supports_variable(variable)` and `validate(test_case,
   variable, year)` → `ValidatorResult`.
4. Add an entry to the table above and document install steps,
   coverage, and fragility points.
5. Mark any tests that hit the network with
   `@pytest.mark.integration` and any that are slow with
   `@pytest.mark.slow`. See `pyproject.toml` for marker definitions.
