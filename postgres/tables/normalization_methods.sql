/*
 Define table to store information about possible normalization methods
 */
-- If needed, drop table to start
-- leave this commented out unless you know what you are doing
-- DROP TABLE IF EXISTS normalization_methods;
CREATE TABLE normalization_methods (
    -- unique serial number to identify the method
    id BIGINT GENERATED ALWAYS AS IDENTITY,
    -- the normalization method used
    method VARCHAR(32) NOT NULL,
    -- notes about the normalization method
    notes TEXT DEFAULT NULL,
    -- add the id as a primary key
    PRIMARY KEY (id),
    -- add a unique constraint
    UNIQUE (method)
);