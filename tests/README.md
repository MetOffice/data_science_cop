# Tests

This directory contains an environment-validation framework and its associated tests.

It currently contains one implemented test (ClimateZones Data Exploration). Additional notebook-
derived tests are expected to be added in future.

## What this is

These are not unit tests of code correctness. Each test runs a reduced, representative workflow derived from a tutorial notebook against a real environment and reports whether it executed successfully.

The goal is to catch environment-breaking regressions (missing packages, broken imports, incompatible
library versions) before they affect users of the actual tutorials, not to verify scientific or
numerical correctness of results.

## Why it exists

These tests provide a lightweight alternative to repeatedly manually running full tutorial notebooks when
validating an environment.

Each test exercises a small but representative workflow, allowing environment changes and updates to
be checked quickly and eventually automated through systems such as Cylc.

## Framework vs test

The framework is designed to support multiple tests, each built around a different notebook-derived
workflow.

Framework-level functionality includes:

- Result classification
- Retention
- Provenance and hashing
- Logging controls
- Wrapper execution support

Each test contributes only its workflow-specific content, configuration and data access logic.

## How to run

Run the wrapper script from `tests/scripts/`:

```bash
./run_test_climatezones_data_exploration.sh
```

Options:

| Flag | Purpose | Default |
|---|---|---|
| `--module <module>` | Environment module to load before running the test | `scitools/community/ml` |
| `--retention` | Save figures, metadata, and artefacts from the run | off |
| `--log-level <level>` | Logging verbosity (e.g. `DEBUG`, `INFO`, `WARNING`, `RESULT`) | `INFO` |

The wrapper loads the specified environment module, invokes the Python test script, and propagates
its exit code unchanged.

## Validation artefacts, provenance and hashing

### Modules

`--module <module>` allows the same representative workflow to be used when validating different environment modules.

For example:

`scitools/community/ml` 

### Log levels

Standard Python [logging levels](https://docs.python.org/3/library/logging.html)
are supported. `RESULT` is a framework-specific level intended to
emphasise validation outcomes.

### Provenance

Provenance information records what was actually executed during a run, including:

- environment module identity
- git version
- script and wrapper cleanliness
- environment inventory information
- environment hash

### Environment inventory and hash

Environment inventory information records the package inventory observed
during the run (can be retained in retention mode).

The environment hash provides a fingerprint of that inventory and can be
used to determine whether runs were performed in the same effective
environment.

### Hashing

Several hashes are generated to provide fingerprints of important run artefacts:

- environment hash
- Python output hash
- individual artefact hashes
- master hash

The master hash provides a single fingerprint derived from the retained
run artefacts (e.g. retained figures, metadata and associated artefact
hashes).

It does not include environment inventory information.

The master hash is only generated when retention mode is enabled.

Hash generation exists to support future comparison workflows.

### Retention

When `--retention` is enabled, run artefacts are saved under `tests/test_run_logs/`.

These may include:

- rendered figures
- metadata
- Python output
- generated hashes
- environment inventory information

Retention mode preserves outputs from the run for later inspection. This is particularly useful when investigating validation failures
or when comparing runs across environment versions.

Retention is best-effort functionality. Failure to retain artefacts should not affect the validation
result itself, which remains authoritative.

## Interpreting results

A test reports one of two outcomes:

- VALIDATED — the reduced representative workflow executed to completion without raising an exception.

- NOT SUCCESSFULLY VALIDATED — an exception was raised while executing the reduced representative workflow.

**What VALIDATED does and does not mean:** VALIDATED means the workflow ran without error in this
environment. It does not verify that the output is scientifically or numerically correct, and it does
not guarantee equivalence with any previous version's output. A workflow can execute cleanly and still
produce subtly wrong results — this is a known limitation of exception-driven validation, not a defect
in the test.

A NOT SUCCESSFULLY VALIDATED result does not by itself indicate that the environment is broken.

Failure classifications are intended to provide an initial indication of whether the reported
exception appears likely to be environment-related, unlikely to be environment-related, or cannot be judged from the exception type alone.

- `LIKELY ENVIRONMENT FAILURE` — e.g. missing/broken imports
- `UNCLEAR WHETHER ENVIRONMENT FAILURE` — e.g. permission or memory errors, which may or may not be
  environment-related
- `LIKELY NON-ENVIRONMENT FAILURE` — everything else (default), including data/file issues that are
  more likely a data or configuration problem than an environment problem

**Detection vs diagnosis:** these tests are designed to reliably detect that something broke, not to
fully diagnose why. When a test fails, retained outputs (see below) are meant to make
troubleshooting easier, not to replace it.

## Current status and future direction

- One test is currently implemented (ClimateZones Data Exploration).
- More notebook-derived tests are expected to be added, reusing the framework-level functionality
  described above.
- Automated scheduling via Cylc is expected in future.
- This README will be updated as the framework evolves; it intentionally documents today's framework
  rather than a future ecosystem that does not yet exist.
