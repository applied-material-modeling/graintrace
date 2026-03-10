[Tensors]
    [a]
        type = Scalar
        values = '1.0'
    []
    [sdirs]
        type = MillerIndex #FillMillerIndex
        values = '1 1 0'
    []
    [splanes]
        type = MillerIndex #FillMillerIndex
        values = '1 1 1'
    []
[]

[Schedulers]
  [simple]
    type = SimpleScheduler
    batch_size = ${device_neml2_batch}
    device = ${device_neml2}
  []
  [hybrid]
    type = StaticHybridScheduler
    batch_sizes = "${nbatchdevice1} ${nbatchdevice2}"
    devices = ${hybrid_devices}
    # trace_file = "hybrid_scheduler_trace.json"
  []
  [simple_MPI]
    type = SimpleMPIScheduler
    batch_sizes = "${nbatchdevice1} ${nbatchdevice2}"
    devices = ${hybrid_devices}
  []
[]

[Data]
    [crystal_geometry]
        type = CubicCrystal
        lattice_parameter = "a"
        slip_directions = "sdirs"
        slip_planes = "splanes"
    []
[]

[Solvers]
    [newton]
        type = NewtonWithLineSearch
        max_linesearch_iterations = 5
        # verbose = true
    []
[]

[Models]
    [spatial_velocity_gradient]
        type = R2IncrementToRate
        variable = 'forces/spatial_velocity_increment'
        time = 'forces/t'
        rate = 'forces/spatial_velocity_gradient'
    []
    [split_to_deformation_rate]
        type = R2toSR2
        input = 'forces/spatial_velocity_gradient'
        output = 'forces/deformation_rate'
    []
    [split_to_vorticity]
        type = R2toWR2
        input = 'forces/spatial_velocity_gradient'
        output = 'forces/vorticity'
    []
    [euler_rodrigues]
        type = RotationMatrix
        from = 'state/orientation'
        to = 'state/orientation_matrix'
    []
    [elastic_tensor]
        type = CubicElasticityTensor
        coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO SHEAR_MODULUS'
        coefficients = '${E} ${nu} ${G}'
    []
    [elasticity]
        type = GeneralElasticity
        elastic_stiffness_tensor = 'elastic_tensor'
        strain = 'state/elastic_strain'
        stress = 'state/internal/cauchy_stress'
    []
    [resolved_shear]
        type = ResolvedShear
    []
    [elastic_stretch]
        type = ElasticStrainRate
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
        n = '${pow_slip_n}'
        gamma0 = '${pow_slip_g0}'
    []
    [slip_strength]
        type = SingleSlipStrengthMap
        constant_strength =' ${slip_constant_strength}'
    []
    [voce_hardening]
        type = VoceSingleSlipHardeningRule
        initial_slope = '${vh_slope_init}'
        saturated_hardening = '${vh_hardening_sat}'
    []
    [integrate_slip_hardening]
        type = ScalarBackwardEulerTimeIntegration
        variable = 'state/internal/slip_hardening'
    []
    [integrate_elastic_strain]
        type = SR2BackwardEulerTimeIntegration
        variable = 'state/elastic_strain'
    []
    [integrate_orientation]
        type = WR2ImplicitExponentialTimeIntegration
        variable = 'state/orientation'
    []
    [implicit_rate]
        type = ComposedModel
        models = "  spatial_velocity_gradient split_to_deformation_rate split_to_vorticity
                    euler_rodrigues elasticity orientation_rate resolved_shear
                    elastic_stretch plastic_deformation_rate plastic_spin
                    sum_slip_rates slip_rule slip_strength voce_hardening
                    integrate_slip_hardening integrate_elastic_strain integrate_orientation"
    []
    [model_without_stress]
        type = ImplicitUpdate
        implicit_model = 'implicit_rate'
        solver = 'newton'
    []
    [full_stress]
        type = SR2toR2
        input = 'state/internal/cauchy_stress'
        output = 'state/internal/full_cauchy_stress'
    []
    ## Post process
    [rotation_matrix]
        type = RotationMatrix
        from = 'state/orientation'
        to = 'state/orientation_matrix'
    []
    [ee_R2]
        type = SR2toR2
        input = 'state/elastic_strain'
        output = 'state/ee_R2'
    []
    [Ree]
        type = R2Multiplication
        A = 'state/orientation_matrix'
        B = 'state/ee_R2'
        to = 'state/Ree'
    []
    [Fe]
        type = R2LinearCombination
        from_var = 'state/Ree state/orientation_matrix'
        to_var = 'state/Fe'
    []
    [Fe_model]
        type = ComposedModel
        models = 'rotation_matrix ee_R2 Ree Fe'
    []
    ## final model
    [model]
        type = ComposedModel
        models = 'model_without_stress full_stress elasticity Fe_model'
        additional_outputs = 'state/elastic_strain state/orientation'
    []
[]