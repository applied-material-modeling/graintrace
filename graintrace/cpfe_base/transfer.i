[MultiApps]
    [regular_grid]
        type = TransientMultiApp
        input_files = 'grid_file.i'
        cli_args = 'base_folder=${base_folder};grid_nx=${grid_nx};grid_ny=${grid_ny};grid_nz=${grid_nz};grid_min_x=${grid_xmin};grid_max_x=${grid_xmax};grid_min_y=${grid_ymin};grid_max_y=${grid_ymax};grid_min_z=${grid_zmin};grid_max_z=${grid_zmax}'
        catch_up = false
        execute_on = '${grid_transfer_execute_on}'
    []
[]

[Transfers]
    [disp_transfer]
        type = MultiAppGeneralFieldShapeEvaluationTransfer
        source_variable = 'disp_x disp_y disp_z'
        variable = 'disp_x disp_y disp_z'
        to_multi_app = 'regular_grid'
        error_on_miss = false
        displaced_source_mesh = false
        displaced_target_mesh = false
    []
    [ori_rodrigues_transfer]
        type = MultiAppGeneralFieldShapeEvaluationTransfer
        source_variable = 'ori_rodrigues_x ori_rodrigues_y ori_rodrigues_z'
        variable = 'ori_rodrigues_x ori_rodrigues_y ori_rodrigues_z'
        to_multi_app = 'regular_grid'
        error_on_miss = false
        displaced_source_mesh = false
        displaced_target_mesh = false
    []
    [strain_transfer]
        type = MultiAppGeneralFieldShapeEvaluationTransfer
        source_variable = 'strain_xx strain_yy strain_zz strain_yz strain_xz strain_xy'
        variable = 'strain_xx strain_yy strain_zz strain_yz strain_xz strain_xy'
        to_multi_app = 'regular_grid'
        error_on_miss = false
        displaced_source_mesh = false
        displaced_target_mesh = false
    []
    [ee_transfer]
        type = MultiAppGeneralFieldShapeEvaluationTransfer
        source_variable = 'ee_xx ee_yy ee_zz ee_yz ee_xz ee_xy'
        variable = 'ee_xx ee_yy ee_zz ee_yz ee_xz ee_xy'
        to_multi_app = 'regular_grid'
        error_on_miss = false
        displaced_source_mesh = false
        displaced_target_mesh = false
    []
    [nye_tensor_transfer]
        type = MultiAppGeneralFieldShapeEvaluationTransfer
        source_variable = 'nye_tensor_11 nye_tensor_12 nye_tensor_13 nye_tensor_21 nye_tensor_22 nye_tensor_23 nye_tensor_31 nye_tensor_32 nye_tensor_33'
        variable = 'nye_tensor_11 nye_tensor_12 nye_tensor_13 nye_tensor_21 nye_tensor_22 nye_tensor_23 nye_tensor_31 nye_tensor_32 nye_tensor_33'
        to_multi_app = 'regular_grid'
        error_on_miss = false
        displaced_source_mesh = false
        displaced_target_mesh = false
    []
    [cauchy_stress_transfer]
        type = MultiAppGeneralFieldShapeEvaluationTransfer
        source_variable = 'cauchy_stress_xx cauchy_stress_yy cauchy_stress_zz cauchy_stress_yz cauchy_stress_xz cauchy_stress_xy'
        variable = 'cauchy_stress_xx cauchy_stress_yy cauchy_stress_zz cauchy_stress_yz cauchy_stress_xz cauchy_stress_xy'
        to_multi_app = 'regular_grid'
        error_on_miss = false
        displaced_source_mesh = false
        displaced_target_mesh = false
    []
[]