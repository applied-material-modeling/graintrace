########################### READ ORIENTATION AND RESIDUAL STRAIN FROM FILE ###########################
[UserObjects]
    [orientation]
        type = PropertyReadFile
        prop_file_name = ${orientation_file}
        read_type = 'block'
        nblock = ${ncell}
        nprop = 3
        use_zero_based_block_indexing = false
        execute_on = TIMESTEP_BEGIN
    []
    [residual_strain]
        type = PropertyReadFile
        prop_file_name = ${residual_strain_file}
        read_type = 'block'
        nblock = ${ncell}
        nprop = 9
        use_zero_based_block_indexing = false
        execute_on = TIMESTEP_BEGIN
    []
[]

[Functions]
    [rodrigues_x]
        type = PiecewiseConstantFromCSV
        column_number = 0
        read_prop_user_object = 'orientation' 
        read_type = 'block'
    []
    [rodrigues_y]
        type = PiecewiseConstantFromCSV
        column_number = 1
        read_prop_user_object = 'orientation' 
        read_type = 'block'
    []
    [rodrigues_z]
        type = PiecewiseConstantFromCSV
        column_number = 2
        read_prop_user_object = 'orientation' 
        read_type = 'block'
    []
    [eixx]
        type = PiecewiseConstantFromCSV
        column_number = 0
        read_prop_user_object = 'residual_strain' 
        read_type = 'block'
    []
    [eiyy]
        type = PiecewiseConstantFromCSV
        column_number = 4
        read_prop_user_object = 'residual_strain' 
        read_type = 'block'
    []
    [eizz]
        type = PiecewiseConstantFromCSV
        column_number = 8
        read_prop_user_object = 'residual_strain' 
        read_type = 'block'
    []
    [eixy]
        type = PiecewiseConstantFromCSV
        column_number = 1
        read_prop_user_object = 'residual_strain' 
        read_type = 'block'
    []
    [eixz]
        type = PiecewiseConstantFromCSV
        column_number = 2
        read_prop_user_object = 'residual_strain' 
        read_type = 'block'
    []
    [eiyz]
        type = PiecewiseConstantFromCSV
        column_number = 5
        read_prop_user_object = 'residual_strain' 
        read_type = 'block'
    []
    [ramping_init]
        type = ParsedFunction
        expression = 't*${strain_unit_conversion}'
    []
    [eixx_ramp]
        type = CompositeFunction
        functions = 'eixx ramping_init'
    []
    [eiyy_ramp]
        type = CompositeFunction
        functions = 'eiyy ramping_init'
    []
    [eizz_ramp]
        type = CompositeFunction
        functions = 'eizz ramping_init'
    []
    [eixy_ramp]
        type = CompositeFunction
        functions = 'eixy ramping_init'
    []
    [eixz_ramp]
        type = CompositeFunction
        functions = 'eixz ramping_init'
    []
    [eiyz_ramp]
        type = CompositeFunction
        functions = 'eiyz ramping_init'
    []
[]

[Materials]
    [initial_orientation]
        type = GenericFunctionVectorMaterial
        prop_names = 'initial_orientation'
        prop_values = 'rodrigues_x rodrigues_y rodrigues_z'
    []
    [residual_strain]
        type = GenericFunctionRankTwoTensor
        # column major-ordered
        tensor_functions = 'eixx_ramp eixy_ramp eixz_ramp 
                            eixy_ramp eiyy_ramp eiyz_ramp
                            eixz_ramp eiyz_ramp eizz_ramp'
        tensor_name = 'residual_strain'
    []
[]