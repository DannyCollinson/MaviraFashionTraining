/*
 Define table to store information about training runs
 */
-- If needed, drop table to start
-- leave this commented out unless you know what you are doing
-- DROP TABLE IF EXISTS train_runs;
CREATE TABLE train_runs (
    -- unique serial number to identify the training run
    id BIGINT GENERATED ALWAYS AS IDENTITY,
    -- time that the training run was started
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    -- time that the training run was stopped or completed training
    end_time TIMESTAMP WITH TIME ZONE NOT NULL,
    -- the id of the training dataset used
    train_set_id INT NOT NULL,
    -- the id of the validation dataset used
    -- explicitly set the default to null so that we can have
    -- training runs that do not use a validation dataset
    val_set_id INT DEFAULT NULL,
    -- model checkpoint id used to start training
    checkpoint_id INT,
    -- batch size used for training
    train_batch_size INT NOT NULL,
    -- number of epochs scheduled for training
    total_epochs_planned INT DEFAULT NULL,
    -- number of warmup epochs scheduled at the start of training
    warmup_epochs INT DEFAULT NULL,
    -- number of epochs that were completed in the training run
    epochs_completed INT NOT NULL,
    -- number of steps that were completed in the training run
    steps_completed INT NOT NULL,
    -- the id of the (first) optimizer used for training
    optimizer_id INT NOT NULL,
    --the id of a second optimizer used for training
    -- explicitly set the default to null so that we can have
    -- training runs that do not use a second optimizer
    optimizer_id_2 INT DEFAULT NULL,
    -- the id of a third optimizer used for training
    optimizer_id_3 INT DEFAULT NULL,
    -- the id of a (first) learning rate scheduler used for training
    lr_scheduler_id INT DEFAULT NULL,
    -- the id of a second learning rate scheduler used for training
    lr_scheduler_id_2 INT DEFAULT NULL,
    -- the id of a third learning rate scheduler used for training
    lr_scheduler_id_3 INT DEFAULT NULL,
    -- the id of the (first) loss function used for training
    loss_function_id INT NOT NULL,
    -- the id of a second loss function used for training
    loss_function_id_2 INT DEFAULT NULL,
    -- the id of a third loss function used for training
    loss_function_id_3 INT DEFAULT NULL,
    -- set the training run id as the primary key
    -- notes about the training run
    notes TEXT DEFAULT NULL,
    -- set the training run id as the primary key
    PRIMARY KEY (id),
    -- set the dataset ids, checkpoint id, optimizer ids, and lr scheduler ids
    -- to be foreign keys
    FOREIGN KEY (train_set_id) REFERENCES datasets (id),
    FOREIGN KEY (val_set_id) REFERENCES datasets (id),
    FOREIGN KEY (checkpoint_id) REFERENCES checkpoints (id),
    FOREIGN KEY (optimizer_id) REFERENCES optimizers (id),
    FOREIGN KEY (optimizer_id_2) REFERENCES optimizers (id),
    FOREIGN KEY (optimizer_id_3) REFERENCES optimizers (id),
    FOREIGN KEY (lr_scheduler_id) REFERENCES lr_schedulers (id),
    FOREIGN KEY (lr_scheduler_id_2) REFERENCES lr_schedulers (id),
    FOREIGN KEY (lr_scheduler_id_3) REFERENCES lr_schedulers (id),
    FOREIGN KEY (loss_function_id) REFERENCES loss_functions (id),
    FOREIGN KEY (loss_function_id_2) REFERENCES loss_functions (id),
    FOREIGN KEY (loss_function_id_3) REFERENCES loss_functions (id)
);