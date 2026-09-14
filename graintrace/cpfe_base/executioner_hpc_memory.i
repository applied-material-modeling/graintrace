[Executioner]
    type = Transient
    solve_type = NEWTON
    automatic_scaling = true

    petsc_options_iname = '-ksp_type -ksp_gmres_restart -pc_type -pc_gamg_type -pc_gamg_threshold -pc_gamg_aggressive_square_graph -pc_gamg_aggressive_coarsening -pc_gamg_coarse_eq_limit -pc_gamg_process_eq_limit -mg_levels_ksp_type -mg_levels_ksp_max_it -mg_levels_pc_type -mg_coarse_ksp_type -mg_coarse_pc_type -mg_coarse_pc_factor_mat_solver_type'
    petsc_options_value = 'fgmres 50 gamg agg 0.2 true 1 1000 5000 chebyshev 1 sor preonly lu superlu_dist'

    residual_and_jacobian_together = false

    line_search = none

    l_max_its = 100
    l_tol = 1e-3
    nl_max_its = 20
    nl_rel_tol = 1e-5
    nl_abs_tol = 1e-6

    end_time = ${total_time}
    dtmax = '${fparse 1000*dt}'

    [TimeStepper]
        type = IterationAdaptiveDT
        dt = ${dt} #s
        optimal_iterations = 7
        iteration_window = 2
        cutback_factor = 0.2
        cutback_factor_at_failure = 0.1
        growth_factor = 2
        linear_iteration_ratio = 1000
    []

    [Predictor]
        type = SimplePredictor
        scale = 1.0
        skip_after_failed_timestep = true
    []
[]
