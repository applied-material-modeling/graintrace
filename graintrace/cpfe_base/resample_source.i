# GridResampler sub-app: load CPFE field data from a saved Exodus at one timestep
# (${cpfe_timestep}) so the parent (resample_grid.i) can shape-evaluate it onto a
# regular grid. No solve; variables are populated purely by the Exodus restart copy.
[Mesh]
    [fmg]
        type = FileMeshGenerator
        file = ${cpfe_exodus}
        use_for_exodus_restart = true
    []
[]

[Problem]
    solve = false
[]

[Executioner]
    type = Transient
    num_steps = 1
    dt = 1
[]

[AuxVariables]
    [ori_rodrigues_x]
        order = FIRST
        family = LAGRANGE
        initial_from_file_var = ori_rodrigues_x
        initial_from_file_timestep = ${cpfe_timestep}
    []
    [ori_rodrigues_y]
        order = FIRST
        family = LAGRANGE
        initial_from_file_var = ori_rodrigues_y
        initial_from_file_timestep = ${cpfe_timestep}
    []
    [ori_rodrigues_z]
        order = FIRST
        family = LAGRANGE
        initial_from_file_var = ori_rodrigues_z
        initial_from_file_timestep = ${cpfe_timestep}
    []
    [strain_xx]
        order = FIRST
        family = LAGRANGE
        initial_from_file_var = strain_xx
        initial_from_file_timestep = ${cpfe_timestep}
    []
    [strain_yy]
        order = FIRST
        family = LAGRANGE
        initial_from_file_var = strain_yy
        initial_from_file_timestep = ${cpfe_timestep}
    []
    [strain_zz]
        order = FIRST
        family = LAGRANGE
        initial_from_file_var = strain_zz
        initial_from_file_timestep = ${cpfe_timestep}
    []
    [strain_yz]
        order = FIRST
        family = LAGRANGE
        initial_from_file_var = strain_yz
        initial_from_file_timestep = ${cpfe_timestep}
    []
    [strain_xz]
        order = FIRST
        family = LAGRANGE
        initial_from_file_var = strain_xz
        initial_from_file_timestep = ${cpfe_timestep}
    []
    [strain_xy]
        order = FIRST
        family = LAGRANGE
        initial_from_file_var = strain_xy
        initial_from_file_timestep = ${cpfe_timestep}
    []
    [ee_xx]
        order = FIRST
        family = LAGRANGE
        initial_from_file_var = ee_xx
        initial_from_file_timestep = ${cpfe_timestep}
    []
    [ee_yy]
        order = FIRST
        family = LAGRANGE
        initial_from_file_var = ee_yy
        initial_from_file_timestep = ${cpfe_timestep}
    []
    [ee_zz]
        order = FIRST
        family = LAGRANGE
        initial_from_file_var = ee_zz
        initial_from_file_timestep = ${cpfe_timestep}
    []
    [ee_yz]
        order = FIRST
        family = LAGRANGE
        initial_from_file_var = ee_yz
        initial_from_file_timestep = ${cpfe_timestep}
    []
    [ee_xz]
        order = FIRST
        family = LAGRANGE
        initial_from_file_var = ee_xz
        initial_from_file_timestep = ${cpfe_timestep}
    []
    [ee_xy]
        order = FIRST
        family = LAGRANGE
        initial_from_file_var = ee_xy
        initial_from_file_timestep = ${cpfe_timestep}
    []
    [nye_tensor_11]
        order = FIRST
        family = LAGRANGE
        initial_from_file_var = nye_tensor_11
        initial_from_file_timestep = ${cpfe_timestep}
    []
    [nye_tensor_12]
        order = FIRST
        family = LAGRANGE
        initial_from_file_var = nye_tensor_12
        initial_from_file_timestep = ${cpfe_timestep}
    []
    [nye_tensor_13]
        order = FIRST
        family = LAGRANGE
        initial_from_file_var = nye_tensor_13
        initial_from_file_timestep = ${cpfe_timestep}
    []
    [nye_tensor_21]
        order = FIRST
        family = LAGRANGE
        initial_from_file_var = nye_tensor_21
        initial_from_file_timestep = ${cpfe_timestep}
    []
    [nye_tensor_22]
        order = FIRST
        family = LAGRANGE
        initial_from_file_var = nye_tensor_22
        initial_from_file_timestep = ${cpfe_timestep}
    []
    [nye_tensor_23]
        order = FIRST
        family = LAGRANGE
        initial_from_file_var = nye_tensor_23
        initial_from_file_timestep = ${cpfe_timestep}
    []
    [nye_tensor_31]
        order = FIRST
        family = LAGRANGE
        initial_from_file_var = nye_tensor_31
        initial_from_file_timestep = ${cpfe_timestep}
    []
    [nye_tensor_32]
        order = FIRST
        family = LAGRANGE
        initial_from_file_var = nye_tensor_32
        initial_from_file_timestep = ${cpfe_timestep}
    []
    [nye_tensor_33]
        order = FIRST
        family = LAGRANGE
        initial_from_file_var = nye_tensor_33
        initial_from_file_timestep = ${cpfe_timestep}
    []
    [cauchy_stress_xx]
        order = FIRST
        family = LAGRANGE
        initial_from_file_var = cauchy_stress_xx
        initial_from_file_timestep = ${cpfe_timestep}
    []
    [cauchy_stress_yy]
        order = FIRST
        family = LAGRANGE
        initial_from_file_var = cauchy_stress_yy
        initial_from_file_timestep = ${cpfe_timestep}
    []
    [cauchy_stress_zz]
        order = FIRST
        family = LAGRANGE
        initial_from_file_var = cauchy_stress_zz
        initial_from_file_timestep = ${cpfe_timestep}
    []
    [cauchy_stress_yz]
        order = FIRST
        family = LAGRANGE
        initial_from_file_var = cauchy_stress_yz
        initial_from_file_timestep = ${cpfe_timestep}
    []
    [cauchy_stress_xz]
        order = FIRST
        family = LAGRANGE
        initial_from_file_var = cauchy_stress_xz
        initial_from_file_timestep = ${cpfe_timestep}
    []
    [cauchy_stress_xy]
        order = FIRST
        family = LAGRANGE
        initial_from_file_var = cauchy_stress_xy
        initial_from_file_timestep = ${cpfe_timestep}
    []
[]

[Outputs]
    console = false
[]
