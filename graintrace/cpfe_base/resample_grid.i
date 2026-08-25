# GridResampler main app: build the regular grid, pull CPFE fields from the
# resample_source.i sub-app (which loaded a saved CPFE Exodus at ${cpfe_timestep})
# via shape-function-evaluation transfer, and sample to grid_out CSV. No solve.
[Mesh]
    type = GeneratedMesh
    dim = 3
    nx = ${grid_nx}
    ny = ${grid_ny}
    nz = ${grid_nz}
    xmin = ${grid_min_x}
    xmax = ${grid_max_x}
    ymin = ${grid_min_y}
    ymax = ${grid_max_y}
    zmin = ${grid_min_z}
    zmax = ${grid_max_z}
[]

[Problem]
    solve = false
[]

[Executioner]
    type = Transient
    num_steps = 1
    dt = 1
[]

[MultiApps]
    [cpfe_source]
        type = FullSolveMultiApp
        input_files = 'resample_source.i'
        cli_args = 'cpfe_exodus=${cpfe_exodus};cpfe_timestep=${cpfe_timestep}'
        execute_on = 'TIMESTEP_BEGIN'
    []
[]

[Transfers]
    [ori_rodrigues_transfer]
        type = MultiAppGeneralFieldShapeEvaluationTransfer
        from_multi_app = 'cpfe_source'
        source_variable = 'ori_rodrigues_x ori_rodrigues_y ori_rodrigues_z'
        variable = 'ori_rodrigues_x ori_rodrigues_y ori_rodrigues_z'
        error_on_miss = false
        execute_on = 'TIMESTEP_BEGIN'
    []
    [strain_transfer]
        type = MultiAppGeneralFieldShapeEvaluationTransfer
        from_multi_app = 'cpfe_source'
        source_variable = 'strain_xx strain_yy strain_zz strain_yz strain_xz strain_xy'
        variable = 'strain_xx strain_yy strain_zz strain_yz strain_xz strain_xy'
        error_on_miss = false
        execute_on = 'TIMESTEP_BEGIN'
    []
    [ee_transfer]
        type = MultiAppGeneralFieldShapeEvaluationTransfer
        from_multi_app = 'cpfe_source'
        source_variable = 'ee_xx ee_yy ee_zz ee_yz ee_xz ee_xy'
        variable = 'ee_xx ee_yy ee_zz ee_yz ee_xz ee_xy'
        error_on_miss = false
        execute_on = 'TIMESTEP_BEGIN'
    []
    [nye_tensor_transfer]
        type = MultiAppGeneralFieldShapeEvaluationTransfer
        from_multi_app = 'cpfe_source'
        source_variable = 'nye_tensor_11 nye_tensor_12 nye_tensor_13 nye_tensor_21 nye_tensor_22 nye_tensor_23 nye_tensor_31 nye_tensor_32 nye_tensor_33'
        variable = 'nye_tensor_11 nye_tensor_12 nye_tensor_13 nye_tensor_21 nye_tensor_22 nye_tensor_23 nye_tensor_31 nye_tensor_32 nye_tensor_33'
        error_on_miss = false
        execute_on = 'TIMESTEP_BEGIN'
    []
    [cauchy_stress_transfer]
        type = MultiAppGeneralFieldShapeEvaluationTransfer
        from_multi_app = 'cpfe_source'
        source_variable = 'cauchy_stress_xx cauchy_stress_yy cauchy_stress_zz cauchy_stress_yz cauchy_stress_xz cauchy_stress_xy'
        variable = 'cauchy_stress_xx cauchy_stress_yy cauchy_stress_zz cauchy_stress_yz cauchy_stress_xz cauchy_stress_xy'
        error_on_miss = false
        execute_on = 'TIMESTEP_BEGIN'
    []
[]

[VectorPostprocessors]
    [element_centroid]
        type = ElementValueSampler
        sort_by = id
        execute_on = 'TIMESTEP_END'
        variable = 'ee_xx ee_yy ee_zz ee_yz ee_xz ee_xy
                    ori_rodrigues_x ori_rodrigues_y ori_rodrigues_z
                    strain_xx strain_yy strain_zz strain_yz strain_xz strain_xy
                    Fe_11 Fe_12 Fe_13
                    Fe_21 Fe_22 Fe_23
                    Fe_31 Fe_32 Fe_33
                    nye_tensor_11 nye_tensor_12 nye_tensor_13
                    nye_tensor_21 nye_tensor_22 nye_tensor_23
                    nye_tensor_31 nye_tensor_32 nye_tensor_33
                    cauchy_stress_xx cauchy_stress_yy cauchy_stress_zz
                    cauchy_stress_yz cauchy_stress_xz cauchy_stress_xy'
    []
[]

[Outputs]
    [csv]
        type = CSV
        file_base = '${base_folder}/grid_out/out'
        execute_on = 'TIMESTEP_END'
    []
    print_linear_residuals = false
[]

# OUTPUT AuxVariables (MONOMIAL FIRST); Fe_* are declared for CSV-schema parity with
# the online grid output but are not transferred (remain 0), matching grid_file.i.
[AuxVariables]
    [ori_rodrigues_x]
        order = FIRST
        family = MONOMIAL
    []
    [ori_rodrigues_y]
        order = FIRST
        family = MONOMIAL
    []
    [ori_rodrigues_z]
        order = FIRST
        family = MONOMIAL
    []
    [strain_xx]
        order = FIRST
        family = MONOMIAL
    []
    [strain_yy]
        order = FIRST
        family = MONOMIAL
    []
    [strain_zz]
        order = FIRST
        family = MONOMIAL
    []
    [strain_yz]
        order = FIRST
        family = MONOMIAL
    []
    [strain_xz]
        order = FIRST
        family = MONOMIAL
    []
    [strain_xy]
        order = FIRST
        family = MONOMIAL
    []
    [ee_xx]
        order = FIRST
        family = MONOMIAL
    []
    [ee_yy]
        order = FIRST
        family = MONOMIAL
    []
    [ee_zz]
        order = FIRST
        family = MONOMIAL
    []
    [ee_yz]
        order = FIRST
        family = MONOMIAL
    []
    [ee_xz]
        order = FIRST
        family = MONOMIAL
    []
    [ee_xy]
        order = FIRST
        family = MONOMIAL
    []
    [Fe_11]
        order = FIRST
        family = MONOMIAL
    []
    [Fe_12]
        order = FIRST
        family = MONOMIAL
    []
    [Fe_13]
        order = FIRST
        family = MONOMIAL
    []
    [Fe_21]
        order = FIRST
        family = MONOMIAL
    []
    [Fe_22]
        order = FIRST
        family = MONOMIAL
    []
    [Fe_23]
        order = FIRST
        family = MONOMIAL
    []
    [Fe_31]
        order = FIRST
        family = MONOMIAL
    []
    [Fe_32]
        order = FIRST
        family = MONOMIAL
    []
    [Fe_33]
        order = FIRST
        family = MONOMIAL
    []
    [cauchy_stress_xx]
        order = FIRST
        family = MONOMIAL
    []
    [cauchy_stress_yy]
        order = FIRST
        family = MONOMIAL
    []
    [cauchy_stress_zz]
        order = FIRST
        family = MONOMIAL
    []
    [cauchy_stress_yz]
        order = FIRST
        family = MONOMIAL
    []
    [cauchy_stress_xz]
        order = FIRST
        family = MONOMIAL
    []
    [cauchy_stress_xy]
        order = FIRST
        family = MONOMIAL
    []
    [nye_tensor_11]
        order = FIRST
        family = MONOMIAL
    []
    [nye_tensor_12]
        order = FIRST
        family = MONOMIAL
    []
    [nye_tensor_13]
        order = FIRST
        family = MONOMIAL
    []
    [nye_tensor_21]
        order = FIRST
        family = MONOMIAL
    []
    [nye_tensor_22]
        order = FIRST
        family = MONOMIAL
    []
    [nye_tensor_23]
        order = FIRST
        family = MONOMIAL
    []
    [nye_tensor_31]
        order = FIRST
        family = MONOMIAL
    []
    [nye_tensor_32]
        order = FIRST
        family = MONOMIAL
    []
    [nye_tensor_33]
        order = FIRST
        family = MONOMIAL
    []
[]
