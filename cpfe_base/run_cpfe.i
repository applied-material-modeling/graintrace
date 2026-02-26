[GlobalParams]
    displacements = 'disp_x disp_y disp_z'
[]

[Mesh]
    [mesh0]
        type = FileMeshGenerator
        file = ${mesh_file}
    []
    [side_sets]
        type = SideSetsFromNormalsGenerator
        input = 'mesh0'
        new_boundary = 'left right bottom top back front'
        normals = '-1 0  0
                    1 0  0
                    0 -1 0
                    0 1  0
                    0 0 -1
                    0 0  1'
        fixed_normal = true
    []
    [node_sets_fixnode]
        type = ExtraNodesetGenerator
        input = side_sets
        new_boundary = 'fixnode'
        coord = '${fixnode_x} ${fixnode_y} ${fixnode_z}'
        tolerance = 1e-4
    []
    [node_sets_yroll]
        type = ExtraNodesetGenerator
        input = node_sets_fixnode
        new_boundary = 'yrollnode'
        coord = '${yroll_x} ${yroll_y} ${yroll_z}'
        tolerance = 1e-4
    []
[]

[Physics]
    [SolidMechanics]
        [QuasiStatic]
            [all]
                strain = FINITE
                new_system = true
                add_variables = true
                formulation = TOTAL
                eigenstrain_names = 'residual_strain'
                volumetric_locking_correction = ${vol_lock_correction_cond}
                material_output_order = FIRST
                generate_output = 'cauchy_stress_xx  cauchy_stress_yy  cauchy_stress_zz 
                                   cauchy_stress_yz  cauchy_stress_xz  cauchy_stress_xy
                                   strain_xx         strain_yy         strain_zz
                                   strain_yz         strain_xz         strain_xy
                                   vonmises_cauchy_stress'
            []
        []
    []
[]

[NEML2]
    input = 'neml2_cpfe.i'
    cli_args = 'slip_constant_strength=${slip_constant_strength} vh_slope_init=${voce_hardening_initial_slope}
                vh_hardening_sat=${voce_hardening_saturation} pow_slip_n=${power_slip_n}
                pow_slip_g0=${power_slip_g0} E=${elastic_E} nu=${elastic_nu} G=${elastic_G}
                device=${device} device_batch=${device_batch}'
    scheduler = 'simple'
    async_dispatch = false
    [all]
        model = 'model'
        verbose = true

        moose_input_types = 'MATERIAL                          POSTPROCESSOR         POSTPROCESSOR
                             MATERIAL                          MATERIAL              MATERIAL'
        moose_inputs = '     spatial_velocity_increment        time                  time          
                             elastic_strain                    orientation           slip_hardening'
        neml2_inputs = '     forces/spatial_velocity_increment forces/t              old_forces/t
                             old_state/elastic_strain          old_state/orientation old_state/internal/slip_hardening'

        moose_output_types = 'MATERIAL                            MATERIAL                      MATERIAL                  
                              MATERIAL                            MATERIAL'
        moose_outputs = '     neml2_cauchy_stress                 elastic_strain                orientation               
                              slip_hardening                      elastic_deformation_gradient'
        neml2_outputs = '     state/internal/full_cauchy_stress   state/elastic_strain          state/orientation         
                              state/internal/slip_hardening       state/Fe'

        moose_derivative_types = 'MATERIAL'
        moose_derivatives = '     neml2_cauchy_jacobian'
        neml2_derivatives = '     state/internal/full_cauchy_stress forces/spatial_velocity_increment'

        initialize_outputs = '      orientation'
        initialize_output_values = 'initial_orientation'
    []
[]

[Postprocessors]
    [time]
        type = TimePostprocessor
        execute_on = 'INITIAL TIMESTEP_BEGIN'
    []
[]

[Functions]
    [coord_x]
        type = ParsedFunction
        expression = 'x'
    []
    [coord_y]
        type = ParsedFunction
        expression = 'y'
    []
    [coord_z]
        type = ParsedFunction
        expression = 'z'
    []
[]

[Materials]
    [stress_copy]
        type = ComputeLagrangianCauchyCustomStress
        custom_cauchy_stress = 'neml2_cauchy_stress'
        custom_cauchy_jacobian = 'neml2_cauchy_jacobian'
        large_kinematics = true
    []
    [nye_tensor]
        type = CurlR2Material
        a11 = Fe_11
        a12 = Fe_12
        a13 = Fe_13
        a21 = Fe_21
        a22 = Fe_22
        a23 = Fe_23
        a31 = Fe_31
        a32 = Fe_32
        a33 = Fe_33
        scale = ${burger_scale}
        curl_name = 'nye_tensor'    
    []
[]

[Executioner]
    type = Transient
    solve_type = NEWTON
    petsc_options = '-ksp_converged_reason'

    petsc_options_iname = '-pc_type -pc_factor_mat_solver_package -ksp_type'
    petsc_options_value = 'lu superlu_dist gmres'

    automatic_scaling = true

    reuse_preconditioner = true
    reuse_preconditioner_max_linear_its = 20

    residual_and_jacobian_together = true

    line_search = none

    nl_abs_tol = 1e-06
    nl_rel_tol = 1e-08
    nl_max_its = 10

    l_max_its = 100

    end_time = ${total_time}
    dtmax = '${fparse 10*dt}'

    [TimeStepper]
        type = IterationAdaptiveDT
        dt = ${dt} #s
        optimal_iterations = 7
        iteration_window = 2
        cutback_factor = 0.2
        cutback_factor_at_failure = 0.5
        growth_factor = 2
        linear_iteration_ratio = 1000
    []

    [Predictor]
        type = SimplePredictor
        scale = 1.0
        skip_after_failed_timestep = true
    []
[]

########################### OUTPUT ###########################
[Outputs]
    file_base = '${base_folder}/sim_output'
    sync_times = '${sync_times}'
    [out]
        type = Exodus
        file_base = '${base_folder}/sim_output'
        execute_postprocessors_on = 'NONE'
    []
    [console]
        type = Console
        execute_postprocessors_on = 'NONE'
    []
    [csv]
        type = CSV
        file_base = '${base_folder}/out'
    []
    print_linear_residuals = false
[]

[AuxVariables]
    [ori_rodrigues_x]
        order = FIRST 
        family = MONOMIAL
        [AuxKernel]
            type = MaterialRealVectorValueAux
            property = 'orientation'
            component = 0
        []
    []
    [ori_rodrigues_y]
        order = FIRST 
        family = MONOMIAL
        [AuxKernel]
            type = MaterialRealVectorValueAux
            property = 'orientation'
            component = 1
        []
    []
    [ori_rodrigues_z]
        order = FIRST 
        family = MONOMIAL
        [AuxKernel]
            type = MaterialRealVectorValueAux
            property = 'orientation'
            component = 2
        []
    []
    [ee_xx]
        order = FIRST 
        family = MONOMIAL
        [AuxKernel]
            type = MaterialSymmetricRankTwoTensorAux
            property = 'elastic_strain'
            component = 0
        []
    []
    [ee_yy]
        order = FIRST 
        family = MONOMIAL
        [AuxKernel]
            type = MaterialSymmetricRankTwoTensorAux
            property = 'elastic_strain'
            component = 1
        []
    []
    [ee_zz]
        order = FIRST 
        family = MONOMIAL
        [AuxKernel]
            type = MaterialSymmetricRankTwoTensorAux
            property = 'elastic_strain'
            component = 2
        []
    []
    [ee_yz]
        order = FIRST 
        family = MONOMIAL
        [AuxKernel]
            type = MaterialSymmetricRankTwoTensorAux
            property = 'elastic_strain'
            component = 3
        []
    []
    [ee_xz]
        order = FIRST 
        family = MONOMIAL
        [AuxKernel]
            type = MaterialSymmetricRankTwoTensorAux
            property = 'elastic_strain'
            component = 4
        []
    []
    [ee_xy]
        order = FIRST 
        family = MONOMIAL
        [AuxKernel]
            type = MaterialSymmetricRankTwoTensorAux
            property = 'elastic_strain'
            component = 5
        []
    []
    [Fe_11]
        order = FIRST 
        family = MONOMIAL
        [AuxKernel]
            type = MaterialRankTwoTensorAux
            property = 'elastic_deformation_gradient'
            i = 0
            j = 0
        []
    []
    [Fe_12]
        order = FIRST 
        family = MONOMIAL
        [AuxKernel]
            type = MaterialRankTwoTensorAux
            property = 'elastic_deformation_gradient'
            i = 0
            j = 1
        []
    []
    [Fe_13]
        order = FIRST 
        family = MONOMIAL
        [AuxKernel]
            type = MaterialRankTwoTensorAux
            property = 'elastic_deformation_gradient'
            i = 0
            j = 2
        []
    []
    [Fe_21]
        order = FIRST 
        family = MONOMIAL
        [AuxKernel]
            type = MaterialRankTwoTensorAux
            property = 'elastic_deformation_gradient'
            i = 1
            j = 0
        []
    []
    [Fe_22]
        order = FIRST 
        family = MONOMIAL
        [AuxKernel]
            type = MaterialRankTwoTensorAux
            property = 'elastic_deformation_gradient'
            i = 1
            j = 1
        []
    []
    [Fe_23]
        order = FIRST 
        family = MONOMIAL
        [AuxKernel]
            type = MaterialRankTwoTensorAux
            property = 'elastic_deformation_gradient'
            i = 1
            j = 2
        []
    []
    [Fe_31]
        order = FIRST 
        family = MONOMIAL
        [AuxKernel]
            type = MaterialRankTwoTensorAux
            property = 'elastic_deformation_gradient'
            i = 2
            j = 0
        []
    []
    [Fe_32]
        order = FIRST 
        family = MONOMIAL
        [AuxKernel]
            type = MaterialRankTwoTensorAux
            property = 'elastic_deformation_gradient'
            i = 2
            j = 1
        []
    []
    [Fe_33]
        order = FIRST 
        family = MONOMIAL
        [AuxKernel]
            type = MaterialRankTwoTensorAux
            property = 'elastic_deformation_gradient'
            i = 2
            j = 2
        []
    []
    [nye_tensor_11]
        order = FIRST 
        family = MONOMIAL
        [AuxKernel]
            type = MaterialRankTwoTensorAux
            property = 'nye_tensor'
            i = 0
            j = 1
        []
    []
    [nye_tensor_12]
        order = FIRST 
        family = MONOMIAL
        [AuxKernel]
            type = MaterialRankTwoTensorAux
            property = 'nye_tensor'
            i = 0
            j = 1
        []
    []
    [nye_tensor_13]
        order = FIRST 
        family = MONOMIAL
        [AuxKernel]
            type = MaterialRankTwoTensorAux
            property = 'nye_tensor'
            i = 0
            j = 2
        []
    []
    [nye_tensor_21]
        order = FIRST 
        family = MONOMIAL
        [AuxKernel]
            type = MaterialRankTwoTensorAux
            property = 'nye_tensor'
            i = 1
            j = 0
        []
    []
    [nye_tensor_22]
        order = FIRST 
        family = MONOMIAL
        [AuxKernel]
            type = MaterialRankTwoTensorAux
            property = 'nye_tensor'
            i = 1
            j = 1
        []
    []
    [nye_tensor_23]
        order = FIRST 
        family = MONOMIAL
        [AuxKernel]
            type = MaterialRankTwoTensorAux
            property = 'nye_tensor'
            i = 1
            j = 2
        []
    []
    [nye_tensor_31]
        order = FIRST 
        family = MONOMIAL
        [AuxKernel]
            type = MaterialRankTwoTensorAux
            property = 'nye_tensor'
            i = 2
            j = 0
        []
    []
    [nye_tensor_32]
        order = FIRST 
        family = MONOMIAL
        [AuxKernel]
            type = MaterialRankTwoTensorAux
            property = 'nye_tensor'
            i = 2
            j = 1
        []
    []
    [nye_tensor_33]
        order = FIRST 
        family = MONOMIAL
        [AuxKernel]
            type = MaterialRankTwoTensorAux
            property = 'nye_tensor'
            i = 2
            j = 2
        []
    []
    [volume]
        order = CONSTANT
        family = MONOMIAL
        [AuxKernel]
            type = VolumeAux
        []
    []
[]


