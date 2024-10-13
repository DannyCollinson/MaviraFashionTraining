/*
 Define table to store information about model checkpoints
 */
-- If needed, drop table to start
-- leave this commented out unless you know what you are doing
-- DROP TABLE IF EXISTS checkpoints;
CREATE TABLE checkpoints (
    -- unique serial number to identify the checkpoint
    id BIGINT GENERATED ALWAYS AS IDENTITY,
    -- the name of the directory that the checkpoint
    -- is saved to relative to the project directory
    dir VARCHAR(300) NOT NULL,
    -- name of the file that the checkpoint is saved to
    fname VARCHAR(300) NOT NULL,
    -- time that the checkpoint was created
    created TIMESTAMP WITH TIME ZONE NOT NULL,
    -- the id of the training run a model checkpoint is saved from
    train_run_id INT NOT NULL,
    -- total number of epochs that the model has been trained for
    epochs INT,
    -- total number of steps that the model has been trained for
    steps INT NOT NULL,
    -- cumulative time that the model has been trained for in seconds
    train_time INT NOT NULL,
    -- the value of the loss function on the training set for the checkpoint
    train_loss NUMERIC(12, 6) NOT NULL,
    -- the accuracy of the model on the training set for the checkpoint
    train_accuracy NUMERIC(6, 4) NOT NULL,
    -- the value of the loss function on the validation set for the checkpoint
    val_loss NUMERIC(12, 6),
    -- the accuracy of the model on the validation set for the checkpoint
    val_accuracy NUMERIC(6, 4),
    -- notes about the checkpoint
    notes TEXT DEFAULT NULL,
    -- add the id as a primary key
    PRIMARY KEY (id)
    -- can't set the train run id as a foreign key in order to set up tables,
    -- but it should be treated as a foreign key
    -- FOREIGN KEY (train_run_id) REFERENCES train_runs (id)
);