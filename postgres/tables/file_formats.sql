/*
 Define table to store information about possible conversion formats
 */
-- If needed, drop table to start
-- leave this commented out unless you know what you are doing
-- DROP TABLE IF EXISTS file_formats;
CREATE TABLE file_formats (
    -- unique serial number to identify the format
    id BIGINT GENERATED ALWAYS AS IDENTITY,
    -- the file format/extension used in the conversion
    format VARCHAR(8) NOT NULL,
    -- the function used to load the images in the format
    load_func VARCHAR(64) DEFAULT NULL,
    -- the function used to save the images in the format
    save_func VARCHAR(64) DEFAULT NULL,
    -- notes about the conversion format
    notes TEXT DEFAULT NULL,
    -- add the id as a primary key
    PRIMARY KEY (id),
    -- add a unique constraint to allow reference by other tables
    UNIQUE (format)
);