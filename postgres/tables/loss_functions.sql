/*
 Define table to store information about loss functions used in training
 */
-- If needed, drop table to start
-- leave this commented out unless you know what you are doing
-- DROP TABLE IF EXISTS loss_functions;
CREATE TABLE loss_functions (
    -- unique serial number to identify the loss function
    id BIGINT GENERATED ALWAYS AS IDENTITY,
    -- name of the loss function class (e.g. MSELoss, CrossEntropyLoss, etc.)
    -- for custom loss functions, use a descriptive name
    class VARCHAR(100) NOT NULL,
    -- the reduction method of the loss function
    reduction CHAR(9) NOT NULL,
    -- the weights of the loss function (for weighted loss functions)
    weights NUMERIC [] DEFAULT NULL,
    -- ignore_index of the loss function (for CrossEntropyLoss)
    ignore_index INT DEFAULT NULL,
    -- label_smoothing of the loss function (for CrossEntropyLoss)
    label_smoothing NUMERIC(16, 12) DEFAULT NULL,
    -- blank of the loss function (for CTCLoss)
    blank INT DEFAULT NULL,
    -- zero_infinity of the loss function (for CTCLoss)
    zero_infinity BOOLEAN DEFAULT NULL,
    -- log_input of the loss function (for PoissonNLLLoss)
    log_input BOOLEAN DEFAULT NULL,
    -- full of the loss function (for PoissonNLLLoss, GaussianNLLLoss)
    full_loss BOOLEAN DEFAULT NULL,
    -- eps of the loss function (for PoissonNLLLoss, GaussianNLLLoss)
    eps NUMERIC(16, 12) DEFAULT NULL,
    -- log_target of the loss function (for KLDivLoss)
    log_target BOOLEAN DEFAULT NULL,
    -- pos_weight of the loss function (for BCEWithLogitsLoss)
    pos_weight NUMERIC [] DEFAULT NULL,
    -- margin of the loss function
    -- (for MarginRankingLoss, HingeEmbeddingLoss,
    -- TripletMarginLoss, TripletMarginWithDistanceLoss)
    margin NUMERIC(16, 12) DEFAULT NULL,
    -- delta of the loss function (for HuberLoss)
    delta NUMERIC(16, 12) DEFAULT NULL,
    -- beta of the loss function (for SmoothL1Loss)
    beta NUMERIC(16, 12) DEFAULT NULL,
    -- p of the loss function (for MultiMarginLoss, TripletMarginLoss)
    p INT DEFAULT NULL,
    -- swap of the loss function
    -- (for TripletMarginLoss, TripletMarginWithDistanceLoss)
    swap BOOLEAN DEFAULT NULL,
    -- distance_function of the loss function
    -- (for TripletMarginWithDistanceLoss)
    distance_function VARCHAR(100) DEFAULT NULL,
    -- notes about the loss function
    notes TEXT DEFAULT NULL,
    -- set the loss function id as the primary key
    PRIMARY KEY (id),
    -- add a unique constraint accross all other columns
    -- to prevent double registration of the same loss function
    UNIQUE (
        class,
        reduction,
        weights,
        ignore_index,
        label_smoothing,
        blank,
        zero_infinity,
        log_input,
        full_loss,
        eps,
        log_target,
        pos_weight,
        margin,
        delta,
        beta,
        p,
        swap,
        distance_function,
        notes
    )
);