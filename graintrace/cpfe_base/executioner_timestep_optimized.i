[Executioner]
    type = Transient
    solve_type = NEWTON
    petsc_options = '-ksp_converged_reason'

    petsc_options_iname = '-pc_type -pc_factor_mat_solver_package -ksp_type'
    petsc_options_value = 'lu superlu_dist gmres'

    automatic_scaling = true

    reuse_preconditioner = true
    reuse_preconditioner_max_linear_its = 20

    residual_and_jacobian_together = false

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
