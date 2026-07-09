# NEML2 v3 crystal-plasticity SOURCE model for the MOOSE/PUMA CPFE run suite.
# Compiled to an AOTI package by run_cpfe_simulation.py, which rewrites the baked
# material parameters below before compiling. Outputs consumed by MOOSE:
# neml2_stress + its derivative (consistent Jacobian), elastic_strain, orientation
# (neml2 MRP), slip_hardening, elastic_deformation_gradient (Fe for Nye/GND tensor).

[Tensors]
  [a]
    type = Python
    expr = 'Scalar(1.0)'
  []
  [sdirs]
    type = Python
    expr = 'MillerIndex(torch.tensor([1, 1, 0], dtype=torch.int64))'
  []
  [splanes]
    type = Python
    expr = 'MillerIndex(torch.tensor([1, 1, 1], dtype=torch.int64))'
  []
[]

[Data]
  [crystal_geometry]
    type = CubicCrystal
    lattice_parameter = 'a'
    slip_directions = 'sdirs'
    slip_planes = 'splanes'
  []
[]

[Models]
  # Kinematics: deformation-gradient increment -> rate -> split
  [spatial_velocity_gradient]
    type = R2IncrementToRate
    increment = 'spatial_deformation_gradient_increment'
    rate = 'spatial_velocity_gradient'
  []
  [split_to_deformation_rate]
    type = R2ToSR2
    input = 'spatial_velocity_gradient'
    output = 'deformation_rate'
  []
  [split_to_vorticity]
    type = R2ToWR2
    input = 'spatial_velocity_gradient'
    output = 'vorticity'
  []

  # Single-crystal constitutive update (cubic elasticity)
  [euler_rodrigues]
    type = RotationMatrix
    from = 'orientation'
    to = 'orientation_matrix'
  []
  [elastic_tensor]
    type = CubicElasticityTensor
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO SHEAR_MODULUS'
    coefficients = '209016.0 0.307 60355.0'  # @BAKE E nu G
  []
  [elasticity]
    type = GeneralElasticity
    elastic_stiffness_tensor = 'elastic_tensor'
    strain = 'elastic_strain'
    stress = 'cauchy_stress'
  []
  [resolved_shear]
    type = ResolvedShear
    stress = 'cauchy_stress'
  []
  [elastic_stretch]
    type = ElasticStrainRate
    deformation_rate = 'deformation_rate'
  []
  [plastic_spin]
    type = PlasticVorticity
  []
  [plastic_deformation_rate]
    type = PlasticDeformationRate
  []
  [orientation_rate]
    type = OrientationRate
  []
  [sum_slip_rates]
    type = SumSlipRates
  []
  [slip_rule]
    type = PowerLawSlipRule
    n = 20.0  # @BAKE power_slip_n
    gamma0 = 0.0001  # @BAKE power_slip_g0
  []
  [slip_strength]
    type = SingleSlipStrengthMap
    constant_strength = 130.0  # @BAKE slip_constant_strength
  []
  [voce_hardening]
    type = VoceSingleSlipHardeningRule
    initial_slope = 1556.09  # @BAKE voce_hardening_initial_slope
    saturated_hardening = 100.0  # @BAKE voce_hardening_saturation
  []
  [integrate_slip_hardening]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'slip_hardening'
  []
  [integrate_elastic_strain]
    type = SR2BackwardEulerTimeIntegration
    variable = 'elastic_strain'
  []
  [integrate_orientation]
    type = WR2ImplicitExponentialTimeIntegration
    variable = 'orientation'
  []
  [implicit_rate]
    type = ComposedModel
    models = "spatial_velocity_gradient split_to_deformation_rate split_to_vorticity
              euler_rodrigues elasticity orientation_rate resolved_shear elastic_stretch
              plastic_deformation_rate plastic_spin sum_slip_rates slip_rule slip_strength
              voce_hardening integrate_slip_hardening integrate_elastic_strain integrate_orientation"
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'implicit_rate'
    unknowns = 'elastic_strain slip_hardening orientation'
  []
[]

[Solvers]
  [newton]
    type = NewtonWithLineSearch
    max_linesearch_iterations = 5
    linear_solver = 'lu'
  []
  [lu]
    type = DenseLU
  []
[]

[Models]
  [predictor]
    type = ConstantExtrapolationPredictor
    unknowns_SR2 = 'elastic_strain'
    unknowns_MRP = 'orientation'
    unknowns_Scalar = 'slip_hardening'
  []
  [model_without_stress]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
    predictor = 'predictor'
  []

  # Post-update quantities exposed to MOOSE
  [full_stress]
    type = SR2ToR2
    input = 'cauchy_stress'
    output = 'neml2_stress'
  []
  # Elastic deformation gradient Fe = R*(I + ee) (for the Nye/GND tensor).
  [rotation_matrix]
    type = RotationMatrix
    from = 'orientation'
    to = 'orientation_matrix'
  []
  [ee_R2]
    type = SR2ToR2
    input = 'elastic_strain'
    output = 'ee_R2'
  []
  [Ree]
    type = R2Multiplication
    A = 'orientation_matrix'
    B = 'ee_R2'
    to = 'Ree'
  []
  [Fe]
    type = R2LinearCombination
    from = 'Ree orientation_matrix'
    to = 'elastic_deformation_gradient'
    weights = '1 1'
  []

  [model]
    type = ComposedModel
    models = 'model_without_stress elasticity full_stress rotation_matrix ee_R2 Ree Fe'
    additional_outputs = 'elastic_strain orientation slip_hardening'
  []
[]
