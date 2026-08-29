#!/usr/bin/env python3
"""Verified FEniCSx copy of the supplied MOOSE Allen--Cahn problem.

The code solves the fully implicit Euler residual and its exact discrete
tangent with respect to reaction_rate. Mobility and kappa remain fixed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

RESULTS = Path(__file__).resolve().parent
CACHE = RESULTS / ".cache"
CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE))
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpi4py import MPI
import mpi4py
import numpy as np
from petsc4py import PETSc
import petsc4py
from dolfinx import fem, io, mesh
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
import dolfinx
import ufl


SOURCE = Path("/home/thanz/moose_projects/case_16_allen_cahn/16_allen_cahn_moose_corrected.i")
PROBES = ((0.25, 0.25), (0.50, 0.50), (0.75, 0.25))


def initial_phi(x):
    return 0.02 * (np.sin(2*np.pi*x[0]) + 0.7*np.cos(2*np.pi*x[1])
                   + 0.4*np.sin(2*np.pi*(x[0] + x[1])))


def assemble_mat(form):
    A = assemble_matrix(fem.form(form), bcs=[])
    A.assemble()
    return A


def make_ksp(comm):
    ksp = PETSc.KSP().create(comm)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    return ksp


def global_l2(domain, f):
    val = fem.assemble_scalar(fem.form(f*f*ufl.dx))
    return float(np.sqrt(domain.comm.allreduce(float(val), op=MPI.SUM)))


def statistics(domain, f):
    owned = f.function_space.dofmap.index_map.size_local
    vals = f.x.array[:owned].real
    lo = domain.comm.allreduce(float(vals.min()), op=MPI.MIN)
    hi = domain.comm.allreduce(float(vals.max()), op=MPI.MAX)
    avg_local = fem.assemble_scalar(fem.form(f*ufl.dx))
    avg = domain.comm.allreduce(float(avg_local), op=MPI.SUM)  # unit area
    return lo, avg, hi


def nearest_values(V, f):
    xy = V.tabulate_dof_coordinates()[:, :2]
    owned = V.dofmap.index_map.size_local
    result = []
    for point in PROBES:
        d2 = np.sum((xy[:owned] - np.asarray(point))**2, axis=1)
        i = int(np.argmin(d2))
        candidates = V.mesh.comm.gather((float(d2[i]), float(f.x.array[i])), root=0)
        if V.mesh.comm.rank == 0:
            result.append(min(candidates, key=lambda item: item[0])[1])
    return result if V.mesh.comm.rank == 0 else None


def solve_case(nx, dt, end_time, reaction_rate, mobility, kappa, save_every,
               output=False, tangent=True, newton_rtol=1e-9,
               newton_atol=1e-11, newton_max_it=30):
    comm = MPI.COMM_WORLD
    steps = int(round(end_time/dt))
    if abs(steps*dt-end_time) > 1e-14:
        raise ValueError("end_time must be an integer multiple of dt")
    domain = mesh.create_unit_square(comm, nx, nx, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", 1))
    old = fem.Function(V, name="phi_old")
    old.interpolate(initial_phi)
    phi = fem.Function(V, name="phi")
    phi.x.array[:] = old.x.array
    sens_old = fem.Function(V, name="dphi_dreaction_rate_old")
    sens = fem.Function(V, name="dphi_dreaction_rate")
    du = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dx = ufl.dx(domain=domain)
    residual = ((phi-old)*v + dt*mobility*kappa*ufl.dot(ufl.grad(phi), ufl.grad(v))
                - dt*mobility*reaction_rate*(phi-phi**3)*v)*dx
    jacobian = ufl.derivative(residual, phi, du)
    delta = fem.Function(V)
    rhs_tangent = (sens_old*v + dt*mobility*(phi-phi**3)*v)*dx
    ksp_newton = make_ksp(comm)
    ksp_tangent = make_ksp(comm)
    times, forward_stats, tangent_stats = [], [], []
    nonlinear_iterations = []

    fw = sn = None
    if output:
        fw = io.XDMFFile(comm, RESULTS/"forward_phi.xdmf", "w")
        sn = io.XDMFFile(comm, RESULTS/"sensitivity_dphi_dreaction_rate.xdmf", "w")
        fw.write_mesh(domain)
        sn.write_mesh(domain)

    def save(t):
        times.append(t)
        forward_stats.append(statistics(domain, phi))
        tangent_stats.append(statistics(domain, sens))
        if fw:
            fw.write_function(phi, t)
            sn.write_function(sens, t)

    save(0.0)
    for step in range(1, steps+1):
        phi.x.array[:] = old.x.array
        phi.x.scatter_forward()
        initial_norm = None
        for it in range(newton_max_it):
            b = assemble_vector(fem.form(-residual))
            b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            norm = float(b.norm())
            if initial_norm is None:
                initial_norm = max(norm, 1e-300)
            if norm <= newton_atol or norm/initial_norm <= newton_rtol:
                break
            A = assemble_mat(jacobian)
            ksp_newton.setOperators(A)
            ksp_newton.solve(b, delta.x.petsc_vec)
            if ksp_newton.getConvergedReason() <= 0:
                raise RuntimeError(f"Newton linear solve failed at step {step}")
            delta.x.scatter_forward()
            phi.x.array[:] += delta.x.array
            phi.x.scatter_forward()
        else:
            raise RuntimeError(f"Newton failed at step {step}: residual={norm:.3e}")
        nonlinear_iterations.append({"step": step, "iterations": it,
                                     "residual": norm,
                                     "relative_residual": norm/initial_norm})
        if tangent:
            A = assemble_mat(jacobian)
            b = assemble_vector(fem.form(rhs_tangent))
            b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            ksp_tangent.setOperators(A)
            ksp_tangent.solve(b, sens.x.petsc_vec)
            if ksp_tangent.getConvergedReason() <= 0:
                raise RuntimeError(f"Tangent solve failed at step {step}")
            sens.x.scatter_forward()
        old.x.array[:] = phi.x.array
        old.x.scatter_forward()
        sens_old.x.array[:] = sens.x.array
        sens_old.x.scatter_forward()
        if step % save_every == 0 or step == steps:
            save(step*dt)
    if fw:
        fw.close()
        sn.close()
    return {"domain": domain, "V": V, "phi": phi, "sens": sens,
            "times": np.asarray(times), "forward_stats": np.asarray(forward_stats),
            "tangent_stats": np.asarray(tangent_stats),
            "nonlinear_iterations": nonlinear_iterations}


def relative_error(a, b, floor=1e-13):
    return abs(a-b)/max(abs(a), abs(b), floor)


def write_statistics(stem, times, values, ylabel):
    with (RESULTS/f"{stem}.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("time", "minimum", "mean", "maximum"))
        writer.writerows(zip(times, *values.T))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, label in enumerate(("minimum", "mean", "maximum")):
        ax.plot(times, values[:, i], label=label)
    ax.set(xlabel="time", ylabel=ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS/f"{stem}.png", dpi=160)
    plt.close(fig)


def link_generic_outputs():
    # XDMF companion HDF5 names are concrete; generic XDMFs reference them.
    for generic, concrete in (("forward_selected_state.xdmf", "forward_phi.xdmf"),
                              ("sensitivity_dselected_state_dselected_control.xdmf",
                               "sensitivity_dphi_dreaction_rate.xdmf")):
        text = (RESULTS/concrete).read_text()
        (RESULTS/generic).write_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=96)
    parser.add_argument("--dt", type=float, default=1e-4)
    parser.add_argument("--end-time", type=float, default=0.1)
    parser.add_argument("--reaction-rate", type=float, default=20.0)
    parser.add_argument("--mobility", type=float, default=1.0)
    parser.add_argument("--kappa", type=float, default=0.05)
    parser.add_argument("--fd-h", type=float, default=2e-3)
    parser.add_argument("--save-every", type=int, default=20)
    args = parser.parse_args()
    comm = MPI.COMM_WORLD
    start = time.time()
    status = "FAIL"
    try:
        baseline = solve_case(args.nx, args.dt, args.end_time, args.reaction_rate,
                              args.mobility, args.kappa, args.save_every, True, True)
        plus = solve_case(args.nx, args.dt, args.end_time, args.reaction_rate+args.fd_h,
                          args.mobility, args.kappa, 10**9, False, False)
        minus = solve_case(args.nx, args.dt, args.end_time, args.reaction_rate-args.fd_h,
                           args.mobility, args.kappa, 10**9, False, False)
        fine = solve_case(args.nx, args.dt/2, args.end_time, args.reaction_rate,
                          args.mobility, args.kappa, 10**9, False, True)
        V = baseline["V"]
        fd = fem.Function(V, name="finite_difference_dphi_dreaction_rate")
        fd.x.array[:] = (plus["phi"].x.array-minus["phi"].x.array)/(2*args.fd_h)
        fd.x.scatter_forward()
        err = fem.Function(V, name="tangent_minus_fd")
        err.x.array[:] = baseline["sens"].x.array-fd.x.array
        err.x.scatter_forward()
        tangent_probe = nearest_values(V, baseline["sens"])
        fd_probe = nearest_values(V, fd)
        phi_probe = nearest_values(V, baseline["phi"])
        fine_phi_probe = nearest_values(fine["V"], fine["phi"])
        fine_sens_probe = nearest_values(fine["V"], fine["sens"])
        tangent_l2 = global_l2(baseline["domain"], baseline["sens"])
        fd_l2 = global_l2(baseline["domain"], fd)
        error_l2 = global_l2(baseline["domain"], err)
        global_rel = error_l2/max(fd_l2, 1e-14)
        init = fem.Function(V)
        init.interpolate(initial_phi)
        init_probe = nearest_values(V, init)
        expected_init = [float(initial_phi(np.array([[x], [y]]))[0]) for x, y in PROBES]
        if comm.rank == 0:
            probes = [{"point": list(p), "tangent": t, "finite_difference": f,
                       "relative_error": relative_error(t, f)}
                      for p, t, f in zip(PROBES, tangent_probe, fd_probe)]
            max_probe_rel = max(row["relative_error"] for row in probes)
            refinement = [{"point": list(p), "phi_dt": a, "phi_dt_over_2": b,
                           "phi_absolute_change": abs(a-b), "sensitivity_dt": c,
                           "sensitivity_dt_over_2": d,
                           "sensitivity_absolute_change": abs(c-d)}
                          for p, a, b, c, d in zip(PROBES, phi_probe, fine_phi_probe,
                                                  tangent_probe, fine_sens_probe)]
            fine_on_coarse_phi = fem.Function(V)
            fine_on_coarse_phi.x.array[:] = fine["phi"].x.array
            fine_on_coarse_sens = fem.Function(V)
            fine_on_coarse_sens.x.array[:] = fine["sens"].x.array
            dphi = fem.Function(V)
            dphi.x.array[:] = fine_on_coarse_phi.x.array-baseline["phi"].x.array
            dsens = fem.Function(V)
            dsens.x.array[:] = fine_on_coarse_sens.x.array-baseline["sens"].x.array
            refinement_global = {"phi_l2_change": global_l2(baseline["domain"], dphi),
                                 "sensitivity_l2_change": global_l2(baseline["domain"], dsens)}
            source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
            all_forward = baseline["forward_stats"]
            all_tangent = baseline["tangent_stats"]
            initial_error = max(abs(a-b) for a, b in zip(init_probe, expected_init))
            acceptance = {"maximum_probe_relative_error_limit": 1e-4,
                          "global_relative_l2_error_limit": 1e-4,
                          "newton_converged_every_step": True}
            passed = max_probe_rel < 1e-4 and global_rel < 1e-4
            status = "PASS" if passed else "FAIL"
            data = {
                "verdict": status,
                "mathematical_contract": {"state": "phi", "control": "reaction_rate",
                    "control_baseline": args.reaction_rate, "control_bounds": None,
                    "held_fixed": {"mobility": args.mobility, "kappa": args.kappa},
                    "domain": "[0,1] x [0,1]", "boundary": "homogeneous natural Neumann",
                    "time_scheme": "fully implicit Euler"},
                "configuration": vars(args), "source": str(SOURCE), "source_sha256": source_hash,
                "finite_difference": {"method": "central", "h": args.fd_h,
                    "minus": args.reaction_rate-args.fd_h,
                    "plus": args.reaction_rate+args.fd_h,
                    "bounds_note": "No physical bounds supplied; both perturbations are positive.",
                    "probes": probes, "maximum_probe_relative_error": max_probe_rel,
                    "tangent_l2": tangent_l2, "finite_difference_l2": fd_l2,
                    "error_l2": error_l2, "global_relative_l2_error": global_rel},
                "initial_condition": {"expected_at_probes": expected_init,
                    "actual_at_probes": init_probe, "maximum_error": initial_error},
                "boundary_condition_audit": "No exterior term or essential BC: zero normal flux is the natural weak-form condition.",
                "field_ranges": {"forward_all_saved": [float(all_forward[:,0].min()),
                    float(all_forward[:,2].max())], "final_forward": all_forward[-1].tolist(),
                    "sensitivity_all_saved": [float(all_tangent[:,0].min()),
                    float(all_tangent[:,2].max())], "constraint_note": "No physical phi bounds were supplied."},
                "time_step_refinement": {"coarse_dt": args.dt, "fine_dt": args.dt/2,
                    "probes": refinement, "global": refinement_global},
                "nonlinear_convergence": {"maximum_iterations": max(x["iterations"] for x in baseline["nonlinear_iterations"]),
                    "maximum_final_residual": max(x["residual"] for x in baseline["nonlinear_iterations"])},
                "acceptance": acceptance, "runtime_seconds": time.time()-start}
            (RESULTS/"validation.json").write_text(json.dumps(data, indent=2)+"\n")
            with (RESULTS/"fd_metrics.txt").open("w") as handle:
                handle.write(f"control = reaction_rate\nbaseline = {args.reaction_rate:.16g}\n")
                handle.write(f"central finite-difference h = {args.fd_h:.16g}\n")
                handle.write(f"perturbed controls = {args.reaction_rate-args.fd_h:.16g}, {args.reaction_rate+args.fd_h:.16g}\n")
                for row in probes:
                    handle.write(f"probe {tuple(row['point'])}: tangent={row['tangent']:.16e}, finite_difference={row['finite_difference']:.16e}, relative_error={row['relative_error']:.8e}\n")
                handle.write(f"tangent L2 norm = {tangent_l2:.16e}\nfinite-difference L2 norm = {fd_l2:.16e}\n")
                handle.write(f"error L2 norm = {error_l2:.16e}\nglobal relative L2 error = {global_rel:.8e}\n")
                handle.write(f"maximum probe relative error = {max_probe_rel:.8e}\n")
            write_statistics("forward_phi_statistics", baseline["times"], all_forward, "phi")
            write_statistics("sensitivity_dphi_dreaction_rate_statistics", baseline["times"], all_tangent,
                             "dphi/dreaction_rate")
            link_generic_outputs()
            with (RESULTS/"validation.md").open("w") as handle:
                handle.write("# Validation of FEniCSx Allen–Cahn conversion\n\n")
                handle.write("The selected state is `phi`; the independent control is `reaction_rate`. Mobility=1 and kappa=0.05 are fixed. No physical bounds were supplied. The centered perturbations remain positive.\n\n")
                handle.write(f"At t={args.end_time:g}, the maximum relative error at three interior probes is **{max_probe_rel:.3e}** and the global relative L2 field error is **{global_rel:.3e}**. Acceptance requires each to be below 1e-4.\n\n")
                handle.write("The analytic initial condition agrees at the probes to " + f"{initial_error:.3e}. The homogeneous no-flux boundary condition is imposed naturally by the weak form. No physical range constraint for phi was supplied; the observed saved range is " + f"[{all_forward[:,0].min():.6g}, {all_forward[:,2].max():.6g}].\n\n")
                handle.write("## Time-step refinement\n\n| Probe | phi change | sensitivity change |\n|---|---:|---:|\n")
                for row in refinement:
                    handle.write(f"| {tuple(row['point'])} | {row['phi_absolute_change']:.6e} | {row['sensitivity_absolute_change']:.6e} |\n")
                handle.write(f"\nGlobal L2 changes from dt to dt/2 are {refinement_global['phi_l2_change']:.6e} for phi and {refinement_global['sensitivity_l2_change']:.6e} for the sensitivity.\n\n")
                handle.write(status+"\n")
            versions = {"python": sys.version.replace("\n", " "), "platform": platform.platform(),
                        "dolfinx": dolfinx.__version__, "ufl": ufl.__version__,
                        "petsc4py": petsc4py.__version__, "mpi4py": mpi4py.__version__,
                        "numpy": np.__version__, "matplotlib": matplotlib.__version__}
            with (RESULTS/"run.log").open("w") as handle:
                handle.write("COMMAND: " + " ".join(sys.argv) + "\n")
                handle.write("VERSIONS: " + json.dumps(versions, sort_keys=True) + "\n")
                handle.write(f"ENV XDG_CACHE_HOME={os.environ['XDG_CACHE_HOME']}\nENV MPLCONFIGDIR={os.environ['MPLCONFIGDIR']}\n")
                handle.write("SOLVER: fully implicit Euler; Newton rtol=1e-9 atol=1e-11 max_it=30; PETSc preonly+LU\n")
                handle.write(f"PARAMETERS: nx={args.nx} ny={args.nx} dt={args.dt} end_time={args.end_time} mobility={args.mobility} kappa={args.kappa} reaction_rate={args.reaction_rate}\n")
                handle.write(f"FINITE_DIFFERENCE: central h={args.fd_h} minus={args.reaction_rate-args.fd_h} plus={args.reaction_rate+args.fd_h}\n")
                handle.write(f"CONVERGENCE: maximum_newton_iterations={data['nonlinear_convergence']['maximum_iterations']} maximum_final_residual={data['nonlinear_convergence']['maximum_final_residual']:.16e}\n")
                handle.write(f"FD_MAX_PROBE_RELATIVE_ERROR={max_probe_rel:.16e} FD_GLOBAL_RELATIVE_L2_ERROR={global_rel:.16e}\n")
                handle.write(f"SOURCE_SHA256={source_hash}\nRUNTIME_SECONDS={data['runtime_seconds']:.6f}\nFINAL_EXIT_STATUS=0\nVERDICT={status}\n")
            print(json.dumps({"verdict": status, "max_probe_relative_error": max_probe_rel,
                              "global_relative_l2_error": global_rel,
                              "runtime_seconds": data["runtime_seconds"]}, indent=2))
        comm.barrier()
    except Exception as exc:
        if comm.rank == 0:
            with (RESULTS/"run.log").open("a") as handle:
                handle.write(f"FINAL_EXIT_STATUS=1\nERROR={type(exc).__name__}: {exc}\nVERDICT=FAIL\n")
        raise


if __name__ == "__main__":
    main()
