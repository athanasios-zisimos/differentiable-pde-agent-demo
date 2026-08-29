# Differentiable PDE Agent Demonstration

A reproducible worked example that converts a MOOSE Allen–Cahn PDE input into a separate, validated differentiable FEniCSx solver.

## Agent instructions

- `AGENTS.md` defines project-level operational and scientific rules.
- `.skills/SKILL.md` defines the differentiable-PDE workflow: tangent/AD sensitivity, central finite-difference validation, field export, and verification.
- `PROMPT.md` is the complete request given to Codex.

## Worked example

Input MOOSE solver:

- `examples/allen_cahn_moose/input/16_allen_cahn_moose_corrected.i`

Generated independent FEniCSx solver:

- `examples/allen_cahn_moose/generated/allen_cahn_fenicsx_differentiable.py`

The original MOOSE input remains unchanged.

## PDE and selected sensitivity

The source model is the nondimensional Allen–Cahn equation:

    d(phi)/dt = D Laplacian(phi) + r (phi - phi^3)

with homogeneous natural no-flux boundaries.

Selected state:

- `phi`

Selected physical control:

- `reaction_rate = r`

Computed sensitivity:

    d(phi) / d(reaction_rate)

All other physical parameters were held fixed.

## Results

`examples/allen_cahn_moose/results_differentiable/` contains:

- forward and sensitivity XDMF/HDF5 time series for ParaView;
- PNG plots of spatial minimum, mean, and maximum versus time;
- central finite-difference comparison in `fd_metrics.txt`;
- numerical summary in `validation.md` and `validation.json`;
- reproducibility information in `run.log`.

The validation confirms agreement between the tangent-linear sensitivity and independently computed central finite differences.

## Codex information

See `CODEX_VERSION.txt` and `MODEL_AND_RUN_INFO.md`.

## Reproduction

A compatible FEniCSx environment is required. From the generated solver directory, run the command recorded in:

    examples/allen_cahn_moose/results_differentiable/run.log
