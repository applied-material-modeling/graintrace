[Problem]
    solve = false
[]

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

[Variables]
    [disp_x]
    []
    [disp_y]
    []
    [disp_z]
    []
[]

[VectorPostprocessors]
    [element_centroid]
        type = ElementValueSampler
        sort_by = id
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

[Executioner]
    type = Transient
    dt = 1e10 #limited by master app
[]

[Outputs]
    file_base = '${base_folder}/sim_output_grid'
    exodus = true
    [console]
        type = Console
        execute_postprocessors_on = 'NONE'
    []
    [csv]
        type = CSV
        file_base = '${base_folder}/grid_out/out'
    []
    print_linear_residuals = false
[]   


# OUTPUT
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