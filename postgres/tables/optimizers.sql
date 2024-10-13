/*
 Define table to store information about optimizers used in training
 */
-- If needed, drop table to start
-- leave this commented out unless you know what you are doing
-- DROP TABLE IF EXISTS optimizers;
CREATE TABLE optimizers (
    -- unique serial number to identify the optimizer
    id BIGINT GENERATED ALWAYS AS IDENTITY,
    -- name of the optimizer class (e.g. Adam, SGD, etc.)
    class VARCHAR(100) NOT NULL,
    -- initial learning rate of the optimizer
    initial_lr NUMERIC(20, 17) NOT NULL,
    -- beta1 of the optimizer
    beta1 NUMERIC(16, 12) DEFAULT NULL,
    -- beta2 of the optimizer
    beta2 NUMERIC(16, 12) DEFAULT NULL,
    -- epsilon of the optimizer
    epsilon NUMERIC(16, 12) DEFAULT NULL,
    -- weight decay of the optimizer
    weight_decay NUMERIC(16, 12) DEFAULT NULL,
    -- amsgrad of the optimizer (for Adam, AdamW)
    amsgrad BOOLEAN DEFAULT NULL,
    -- lambd of the optimizer (for ASGD)
    lambd NUMERIC(16, 12) DEFAULT NULL,
    -- alpha of the optimizer (for ASGD, RMSprop)
    alpha NUMERIC(16, 12) DEFAULT NULL,
    -- t0 of the optimizer (for ASGD)
    t0 NUMERIC(16, 12) DEFAULT NULL,
    -- momentum of the optimizer (for RMSprop, SGD)
    momentum NUMERIC(16, 12) DEFAULT NULL,
    -- dampening of the optimizer (for SGD)
    dampening NUMERIC(16, 12) DEFAULT NULL,
    -- nesterov of the optimizer (for SGD)
    nesterov BOOLEAN DEFAULT NULL,
    -- rho of the optimizer (for Adadelta)
    rho NUMERIC(16, 12) DEFAULT NULL,
    -- lr decay of the optimizer (for Adagrad)
    lr_decay NUMERIC(16, 12) DEFAULT NULL,
    -- initial_accumulator_value of the optimizer (for Adagrad)
    initial_accumulator_value NUMERIC(16, 12) DEFAULT NULL,
    -- max_iter of the optimizer (for L-BFGS)
    max_iter INT DEFAULT NULL,
    -- max_eval of the optimizer (for L-BFGS)
    max_eval INT DEFAULT NULL,
    -- tolerance_grad of the optimizer (for L-BFGS)
    tolerance_grad NUMERIC(16, 12) DEFAULT NULL,
    -- tolerance_change of the optimizer (for L-BFGS)
    tolerance_change NUMERIC(16, 12) DEFAULT NULL,
    -- history_size of the optimizer (for L-BFGS)
    history_size INT DEFAULT NULL,
    -- line_search_fn of the optimizer (for L-BFGS)
    line_search_fn VARCHAR(100) DEFAULT NULL,
    -- momentum_decay of the optimizer (for NAdam)
    momentum_decay NUMERIC(16, 12) DEFAULT NULL,
    -- decoupled_weight_decay of the optimizer (for NAdam, RAdam)
    decoupled_weight_decay BOOLEAN DEFAULT NULL,
    -- centered of the optimizer (for RMSprop)
    centered BOOLEAN DEFAULT NULL,
    -- eta1 of the optimizer (for Rprop)
    eta1 NUMERIC(16, 12) DEFAULT NULL,
    -- eta2 of the optimizer (for Rprop)
    eta2 NUMERIC(16, 12) DEFAULT NULL,
    -- step_size1 of the optimizer (for Rprop)
    step_size1 NUMERIC(16, 12) DEFAULT NULL,
    -- step_size2 of the optimizer (for Rprop)
    step_size2 NUMERIC(16, 12) DEFAULT NULL,
    -- notes about the optimizer
    notes TEXT DEFAULT NULL,
    -- set the optimizer id as the primary key
    PRIMARY KEY (id),
    -- add a unique constraint accross all other columns
    -- to prevent double registration of the same optimizer
    UNIQUE (
        class,
        initial_lr,
        beta1,
        beta2,
        epsilon,
        weight_decay,
        amsgrad,
        lambd,
        alpha,
        t0,
        momentum,
        dampening,
        nesterov,
        rho,
        lr_decay,
        initial_accumulator_value,
        max_iter,
        max_eval,
        tolerance_grad,
        tolerance_change,
        history_size,
        line_search_fn,
        momentum_decay,
        decoupled_weight_decay,
        centered,
        eta1,
        eta2,
        step_size1,
        step_size2,
        notes
    )
);