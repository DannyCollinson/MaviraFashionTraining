/*
 Define table to store information about lr schedulers used in training
 */
-- If needed, drop table to start
-- leave this commented out unless you know what you are doing
-- DROP TABLE IF EXISTS lr_schedulers;
CREATE TABLE lr_schedulers (
    -- unique serial number to identify the lr scheduler
    id BIGINT GENERATED ALWAYS AS IDENTITY,
    -- name of the lr scheduler class (e.g. StepLR, ReduceLROnPlateau, etc.)
    class VARCHAR(100) NOT NULL,
    -- last epoch of the lr scheduler
    last_epoch INT DEFAULT NULL,
    -- factor of the lr scheduler (for ConstantLR, ReduceLROnPlateau)
    factor NUMERIC(16, 12) DEFAULT NULL,
    -- patience of the lr scheduler (for ReduceLROnPlateau)
    patience INT DEFAULT NULL,
    -- threshold of the lr scheduler (for ReduceLROnPlateau)
    threshold NUMERIC(16, 12) DEFAULT NULL,
    -- threshold mode of the lr scheduler (for ReduceLROnPlateau)
    threshold_mode CHAR(3) DEFAULT NULL,
    -- cooldown of the lr scheduler (for ReduceLROnPlateau)
    cooldown INT DEFAULT NULL,
    -- min_lr of the lr scheduler (for ReduceLROnPlateau)
    min_lr NUMERIC(16, 12) DEFAULT NULL,
    -- eps of the lr scheduler (for ReduceLROnPlateau)
    eps NUMERIC(16, 12) DEFAULT NULL,
    -- Tmax of the lr scheduler
    -- (for CosineAnnealingLR, CosineAnnealingWarmRestarts)
    Tmax INT DEFAULT NULL,
    -- eta_min of the lr scheduler
    -- (for CosineAnnealingLR, CosineAnnealingWarmRestarts)
    eta_min NUMERIC(16, 12) DEFAULT NULL,
    -- base_lr of the lr scheduler (for CyclicLR)
    base_lr NUMERIC(16, 12) DEFAULT NULL,
    -- max_lr of the lr scheduler (for CyclicLR, OneCycleLR)
    max_lr NUMERIC(16, 12) DEFAULT NULL,
    -- step_size_up of the lr scheduler (for CyclicLR)
    step_size_up INT DEFAULT NULL,
    -- step_size_down of the lr scheduler (for CyclicLR)
    step_size_down INT DEFAULT NULL,
    -- mode of the lr scheduler (for CyclicLR)
    mode CHAR(10) DEFAULT NULL,
    -- scale_fn of the lr scheduler (for CyclicLR)
    scale_fn VARCHAR(100) DEFAULT NULL,
    -- scale_mode of the lr scheduler (for CyclicLR)
    scale_mode CHAR(10) DEFAULT NULL,
    -- cycle_momentum of the lr scheduler (for CyclicLR, OneCycleLR)
    cycle_momentum BOOLEAN DEFAULT NULL,
    -- base_momentum of the lr scheduler (for CyclicLR, OneCycleLR)
    base_momentum NUMERIC(16, 12) DEFAULT NULL,
    -- max_momentum of the lr scheduler (for CyclicLR, OneCycleLR)
    max_momentum NUMERIC(16, 12) DEFAULT NULL,
    -- total_steps of the lr scheduler (for OneCycleLR)
    -- SHOULD BE PLACED IN NOTES
    -- total_steps INT DEFAULT NULL,
    -- pct_start of the lr scheduler (for OneCycleLR)
    -- SHOULD BE PLACED IN NOTES
    -- pct_start NUMERIC(16, 12) DEFAULT NULL,
    -- anneal_strategy of the lr scheduler (for OneCycleLR)
    -- SHOULD BE PLACED IN NOTES
    -- anneal_strategy CHAR(6) DEFAULT NULL,
    -- div_factor of the lr scheduler (for OneCycleLR)
    -- SHOULD BE PLACED IN NOTES
    -- div_factor NUMERIC(16, 12) DEFAULT NULL,
    -- final_div_factor of the lr scheduler (for OneCycleLR)
    -- SHOULD BE PLACED IN NOTES
    -- final_div_factor NUMERIC(16, 12) DEFAULT NULL,
    -- three_phase of the lr scheduler (for OneCycleLR)
    -- SHOULD BE PLACED IN NOTES
    -- three_phase BOOLEAN DEFAULT NULL,
    -- lr lambda function of the lr scheduler (for LambdaLR, MultiplicativeLR)
    -- SHOULD BE PLACED IN NOTES
    -- lr_lambda VARCHAR(300) DEFAULT NULL,
    -- step size of the lr scheduler
    -- (for StepLR, ExponentialLR,
    -- CosineAnnealingLR, CosineAnnealingWarmRestarts)
    step_size INT DEFAULT NULL,
    -- gamma of the lr scheduler
    -- (for StepLR, MultiStepLR, ExponentialLR, CyclicLR)
    gamma NUMERIC(16, 12) DEFAULT NULL,
    -- milestones of the lr scheduler (for MultiStepLR)
    milestones VARCHAR(300) DEFAULT NULL,
    -- total_iters of the lr scheduler
    -- (for ConstantLR, LinearLR, PolynomialLR, CyclicLR)
    total_iters INT DEFAULT NULL,
    -- start_factor of the lr scheduler (for LinearLR, CyclicLR)
    start_factor NUMERIC(16, 12) DEFAULT NULL,
    -- end_factor of the lr scheduler (for LinearLR, CyclicLR)
    end_factor NUMERIC(16, 12) DEFAULT NULL,
    -- power of the lr scheduler (for PolynomialLR)
    power NUMERIC(16, 12) DEFAULT NULL,
    -- T_0 of the lr scheduler (for CosineAnnealingWarmRestarts)
    T_0 INT DEFAULT NULL,
    -- T_mult of the lr scheduler (for CosineAnnealingWarmRestarts)
    T_mult INT DEFAULT NULL,
    -- notes about the lr scheduler
    notes TEXT DEFAULT NULL,
    -- set the lr scheduler id as the primary key
    PRIMARY KEY (id),
    -- add a unique constraint accross all other columns
    -- to prevent double registration of the same lr scheduler
    UNIQUE (
        class,
        last_epoch,
        factor,
        patience,
        threshold,
        threshold_mode,
        cooldown,
        min_lr,
        eps,
        Tmax,
        eta_min,
        base_lr,
        max_lr,
        step_size_up,
        step_size_down,
        mode,
        scale_fn,
        scale_mode,
        cycle_momentum,
        base_momentum,
        max_momentum,
        step_size,
        gamma,
        milestones,
        total_iters,
        start_factor,
        end_factor,
        power,
        T_0,
        T_mult
    )
);