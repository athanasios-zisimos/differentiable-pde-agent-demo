# Differentiable PDE Conversion and Validation

## Purpose

Use this skill when converting a classical finite-element PDE solver into a
differentiable FEniCSx or JAX-FEM solver. It applies to steady, transient,
linear, nonlinear, and coupled systems.

## Skill selection

Read the available files under `.skills/`.

Use the minimum relevant skills:
- pick file `skill.md`;

## 1. Establish the mathematical contract

Identify the residual system

`R(u; theta) = 0`,

where `u` is the selected state field (or coupled vector of fields) and
`theta` is a selected physical control. Confirm the domain, initial conditions,
boundary conditions, units, and the control's location in the residual.

If any of these are unknown and materially affect the solution, record the
ambiguity in `clarification.md` instead of assuming it.

## 2. Compute a full-field sensitivity

For the tangent field `s = du/dtheta`, use the discrete linearized system

`(dR/du) s = -dR/dtheta`.

- **FEniCSx:** Prefer `dolfinx-adjoint` when annotation supports the required
  operation; otherwise assemble and solve the explicit UFL tangent-linear
  system.
- **JAX-FEM:** Prefer `jax.jvp`, `jax.jacrev`, or a correctly defined implicit
  differentiation/VJP rule. Do not differentiate only a local expression when
  the requested state depends on coupled PDE solves.

For transient models, propagate the derivative through every physical time
step and save it at every requested output time.

## 3. Validate independently

At a baseline control `theta`, choose a perturbation `h` that remains inside
the control's physical range and compare

`s_FD = [u(theta + h) - u(theta - h)] / (2 h)`

against the tangent/AD sensitivity. Report values and relative errors at at
least three interior probes and one global norm when feasible. Also record
solver tolerances; numerical residuals should be smaller than the finite-
difference signal.

For transient PDEs, perform a time-step refinement (`dt` versus `dt/2`) when
computationally feasible. For spatially resolved problems, request a mesh
refinement check or explicitly document why it is unavailable.

## 4. Required deliverables

Write these in the declared results directory:

- forward state field time series;
- sensitivity field time series;
- FEniCSx XDMF/HDF5 or JAX VTK/NPZ visualization data;
- PNG time histories of spatial minimum, mean, and maximum for both fields;
- finite-difference metrics and validation summary;
- initial/boundary-condition and physical-range audit;
- `run.log`, including versions, commands, tolerances, and exit status.

## 5. Completion criterion

Declare success only when the generated solver ran, requested outputs exist,
the original source is unchanged, and the sensitivity comparison is reported.
If any item fails, state the limitation plainly rather than claiming a verified
result.
