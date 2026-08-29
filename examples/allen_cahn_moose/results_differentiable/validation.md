# Validation of FEniCSx Allen–Cahn conversion

The selected state is `phi`; the independent control is `reaction_rate`. Mobility=1 and kappa=0.05 are fixed. No physical bounds were supplied. The centered perturbations remain positive.

At t=0.1, the maximum relative error at three interior probes is **3.481e-07** and the global relative L2 field error is **5.867e-07**. Acceptance requires each to be below 1e-4.

The analytic initial condition agrees at the probes to 0.000e+00. The homogeneous no-flux boundary condition is imposed naturally by the weak form. No physical range constraint for phi was supplied; the observed saved range is [-0.164668, 0.24139].

## Time-step refinement

| Probe | phi change | sensitivity change |
|---|---:|---:|
| (0.25, 0.25) | 9.702225e-05 | 1.951825e-05 |
| (0.5, 0.5) | 6.757405e-05 | 1.400473e-05 |
| (0.75, 0.25) | 9.810885e-05 | 1.987793e-05 |

Global L2 changes from dt to dt/2 are 9.283352e-05 for phi and 1.814514e-05 for the sensitivity.

PASS
