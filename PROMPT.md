## Prompt provided to intiate the agent

You are converting the supplied PDE solver into a separate, executable,
differentiable solver in the output format selected by the user before this
request:
- FEniCSx + dolfinx-adjoint/tangent-linear; or
- JAX-FEM + JAX jvp/jacrev.

First read and follow AGENTS.md and SKILL.md in the current work directory,
when those files are present. Read the original supplied solver and any
accompanying mathematical files and keep the SKILL.md file related to the PDE problem of the supplied .py file.

Keep the original solver and frozen inputs unchanged.

Identify the solved PDE state fields and the physical control parameters.
Do not present mesh dimensions, time step, final time, output settings, solver
tolerances, random seeds, or bookkeeping flags as physical controls unless
explicitly requested as a numerical or shape sensitivity.

Present reviewed state and physical-control menus and wait for the user to
select exactly one state and one control.

If governing PDE terms, initial or boundary conditions, units, the requested
derivative, or the parametrization of a field control are genuinely ambiguous,
ask focused clarification questions before changing the mathematical model.

After the user selects a state and control, treat the selected control as an
independent physical parameter and hold all other physical parameters fixed.
If the source uses a combined coefficient, for example

D = mobility * kappa,

state clearly which independent parameter is differentiated and which
parameters remain fixed.

After selection, generate and validate the full field sensitivity:

d(selected_state) / d(selected_control)

over the complete simulation interval used by the source solver.

If the source solver is stationary, do not invent a time-dependent model.
Instead, generate the forward field and its full-field sensitivity for the
stated stationary PDE.

If the selected control is a field or time-dependent prescribed datum rather
than a scalar, define and document the requested derivative direction or
parametrization before implementation. Do not silently replace a field-control
derivative with a scalar derivative.

Create a separate differentiable solver copy in the user-selected output
format.

Preserve the source PDE, geometry, mesh, initial conditions, boundary
conditions, sign conventions, material laws, parameter values, and simulation
interval.

For FEniCSx output, use dolfinx-adjoint where appropriate or an explicit
tangent-linear UFL/FEM formulation.

For JAX-FEM output, use JAX jvp, jacrev, jacfwd, or an equivalent valid
differentiable implicit-solve method.

Compute the complete discrete full-field sensitivity at every saved time,
including every downstream coupled PDE solve.

Output-location contract:

You are working inside the harness-created work/ directory.

Do not write generated files outside the current work/ directory.

Create one dedicated results folder:

results_differentiable/

Save all generated solver copies, fields, figures, tables, validation reports,
and logs inside results_differentiable/.

Required deliverables:

1. Forward selected-state field time series for transient problems, or the
   forward field for stationary problems.

2. Full sensitivity field

   d(selected_state) / d(selected_control)

   at every saved time for transient problems, or for the stationary solution.

3. Visualization field files:

   For FEniCSx output, write separate XDMF/HDF5 time series suitable for
   ParaView:

   - results_differentiable/forward_selected_state.xdmf
   - results_differentiable/sensitivity_dselected_state_dselected_control.xdmf

   Also use concrete names where possible, for example:

   - results_differentiable/forward_phi.xdmf
   - results_differentiable/sensitivity_dphi_dreaction_rate.xdmf

   For JAX-FEM output, write VTK and/or compressed NPZ field time series:
   - results_differentiable/forward_selected_state.vtk or .npz
   - results_differentiable/sensitivity_dselected_state_dselected_control.vtk
     or .npz


4. For transient problems, create PNG plots of spatial mean, minimum, and
   maximum versus time for:

   - selected_state;
   - d(selected_state)/d(selected_control).

5. Validate d(selected_state)/d(selected_control) independently with central
   finite differences at at least three interior probe points:

   [selected_state(selected_control + h)
    - selected_state(selected_control - h)] / (2h).

   Choose and document a finite-difference perturbation h that is small enough
   to approximate the derivative but large enough to exceed solver tolerances
   and numerical noise.

   Keep selected_control - h and selected_control + h inside declared physical
   parameter bounds.

6. Report at every probe:

   - tangent/AD sensitivity;
   - finite-difference sensitivity;
   - relative error;
   - maximum relative error across all probes.

   Also report a global field error, such as a relative L2 error, when feasible.

   Save the machine-readable comparison in:

   results_differentiable/fd_metrics.txt

7. Verify initial and boundary conditions in the forward result.

   Report field ranges and flag violations of stated physical constraints, such
   as non-negative concentration, density, temperature, or damage variables
   where those constraints are part of the model.

8. For transient problems, compare dt with dt/2 when computationally feasible.

   Report changes in selected_state and
   d(selected_state)/d(selected_control).

   For stationary problems, perform a mesh-refinement check when feasible.

   If a time-step or mesh-refinement check is unavailable, explain why in
   validation.md rather than claiming that it was completed.

9. Create:

   - results_differentiable/validation.md
   - results_differentiable/validation.json
   - results_differentiable/run.log

The run.log must record commands, library versions, environment flags, solver
tolerances, convergence information, finite-difference step size, and final
exit status.

End validation.md with exactly one final verdict:

PASS:
The forward solution, full sensitivity calculation, and independent
finite-difference validation completed successfully and satisfy the declared
acceptance criteria.

FAIL:
A required calculation, output, or verification failed.

BLOCKED:
Essential mathematical or physical information is missing and user
clarification is required.

Do not claim success unless the forward run, sensitivity calculation, and
finite-difference validation have actually completed.

Deliver only independently verified results.
