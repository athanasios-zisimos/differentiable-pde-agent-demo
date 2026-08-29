# Verified MOOSE Allen--Cahn input: spatial phase field with no-flux boundaries.
#
# Solves, on the unit square:
#     d(phi)/dt = diffusivity * Laplacian(phi)
#                 + reaction_rate * (phi - phi^3)
#
# Here diffusivity = mobility * kappa.  Natural boundary conditions of the
# weak form give grad(phi) . n = 0 on all exterior boundaries, so no [BCs]
# block is required.

[Mesh]
  type = GeneratedMesh
  dim = 2
  nx = 96
  ny = 96
  xmin = 0.0
  xmax = 1.0
  ymin = 0.0
  ymax = 1.0
[]

[Modules]
  [PhaseField]
    [Nonconserved]
      [phi]
        # The module evolves phi_t = mobility * (kappa * Laplacian(phi)
        #                                      - dF/dphi).
        mobility = 1.0
        kappa = 0.05
        free_energy = F
      []
    []
  []
[]

[ICs]
  [phi_initial]
    type = FunctionIC
    variable = phi
    # Deterministic, smooth perturbation around phi = 0.
    function = phi_initial_function
  []
[]

[Functions]
  [phi_initial_function]
    type = ParsedFunction
    expression = '0.02*(sin(2*pi*x)+0.7*cos(2*pi*y)+0.4*sin(2*pi*(x+y)))'
  []
[]

[Materials]
  [free_energy]
    type = DerivativeParsedMaterial
    property_name = F
    coupled_variables = 'phi'
    # F(phi) = reaction_rate * (phi^4/4 - phi^2/2), hence
    # -dF/dphi = reaction_rate * (phi - phi^3).
    constant_names = 'reaction_rate'
    constant_expressions = '20.0'
    expression = 'reaction_rate*(0.25*phi^4 - 0.5*phi^2)'
    derivative_order = 2
  []
[]

[Executioner]
  type = Transient
  scheme = implicit-euler
  solve_type = NEWTON
  dt = 1.0e-4
  end_time = 0.1
  nl_rel_tol = 1.0e-9
  nl_abs_tol = 1.0e-11
  nl_max_its = 30
  petsc_options_iname = '-pc_type -ksp_type'
  petsc_options_value = 'lu preonly'
[]

[Postprocessors]
  [phi_average]
    type = ElementAverageValue
    variable = phi
    execute_on = 'initial timestep_end'
  []
  [phi_minimum]
    type = NodalExtremeValue
    variable = phi
    value_type = min
    execute_on = 'initial timestep_end'
  []
  [phi_maximum]
    type = NodalExtremeValue
    variable = phi
    value_type = max
    execute_on = 'initial timestep_end'
  []
[]

[Outputs]
  file_base = allen_cahn_moose_corrected_out
  exodus = true
  csv = true
  execute_on = 'initial timestep_end'
[]
