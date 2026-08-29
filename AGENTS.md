# AGENTS.md — Framework-Agnostic PDE Conversion Rules

## Purpose

This project converts supplied PDE solvers into independently verifiable,
differentiable FEniCSx or JAX-FEM solver copies. Preserve the source solver as
evidence; never silently replace it.

## Operational guardrails

1. **Read-only inputs.** Treat `frozen_inputs/` as immutable. Create generated
   code, plots, fields, logs, and temporary files only in `work/` or the
   declared deliverable directory.
2. **Deterministic setup.** Before importing DOLFINx/FFCx, set
   `XDG_CACHE_HOME` to a writable local cache. For JAX, explicitly configure
   `jax_enable_x64=True` when the source problem needs double precision.
3. **Traceability.** Record commands, library versions, environment flags,
   solver tolerances, convergence information, and final exit status in
   `run.log`.
4. **State versus control.** States are solved current-time fields. Controls
   are physical coefficients, material parameters, source amplitudes, or
   prescribed data that actually enter the selected residual. Do not present
   mesh dimensions, time step, final time, output settings, or bookkeeping
   flags as physical controls unless the user explicitly requests numerical or
   shape sensitivity.
5. **No invented physics.** Do not guess missing PDE terms, units, material
   values, initial conditions, or boundary conditions. Write focused questions
   to `clarification.md` and wait for confirmation when those choices change
   the mathematical problem.
6. **Self-contained verification.** Generated solvers must run from the
   command line and must not require an IDE, notebook, or proprietary tool.

## Conversion rules

1. Preserve the governing PDE, initial conditions, boundary conditions, domain,
   and sign conventions from the source of truth.
2. Create a separate solver copy rather than editing the original.
3. For a selected field state `u` and physical control `theta`, compute the
   total discrete sensitivity `du/dtheta`, including every downstream coupled
   solve.
4. Treat the selected control as an independent physical parameter and hold all
   other physical parameters fixed.
5. If the source uses a combined coefficient, such as `D = M*kappa`, state
   explicitly whether the derivative is with respect to `D`, `M`, or `kappa`,
   and state which remaining parameters are fixed.
6. For a field-valued or time-dependent prescribed control, require the user
   to define its parameterization or perturbation direction before computing a
   derivative.
7. For stationary PDEs, do not invent a time-dependent model. Use a
   mesh-refinement check when feasible instead of a time-step refinement.
8. Independently validate the sensitivity with central finite differences when
   feasible. Report probe and global errors, not only a visual comparison.
9. Export forward and sensitivity fields separately, plus numerical evidence
   needed to reproduce and assess the result.
10. End `validation.md` with exactly one verdict: `PASS`, `FAIL`, or `BLOCKED`.
