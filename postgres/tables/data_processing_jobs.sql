/*
 Define table to store information about data processing jobs
 */
-- If needed, drop table to start
-- leave this commented out unless you know what you are doing
-- DROP TABLE IF EXISTS data_processing_jobs;
CREATE TABLE data_processing_jobs (
    -- unique serial number to identify the processing job
    id BIGINT GENERATED ALWAYS AS IDENTITY,
    -- time that the job was started
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    -- time that the job was stopped or completed training
    end_time TIMESTAMP WITH TIME ZONE DEFAULT NULL,
    -- boolean for whether or not filename cleaning is performed
    cleaning BOOLEAN NOT NULL,
    -- boolean for whether or not resizing is performed
    resizing BOOLEAN NOT NULL,
    -- boolean for whether or not train/val/test splitting is performed
    splitting BOOLEAN NOT NULL,
    -- boolean for whether or not normalization is performed
    normalization BOOLEAN NOT NULL,
    -- boolean for whether or not file format conversion is performed
    conversion BOOLEAN NOT NULL,
    -- the --raw argument for the processing job
    raw_path VARCHAR(300) DEFAULT NULL,
    -- the path to the starting dataset
    starting_dataset_path VARCHAR(300) NOT NULL,
    -- the dataset id for the post-cleaning dataset
    cleaned_dataset_id INT DEFAULT NULL,
    -- the dataset id for the post-resizing dataset
    resized_dataset_id INT DEFAULT NULL,
    -- the dataset id for the post-splitting dataset
    split_dataset_id INT DEFAULT NULL,
    -- the dataset id for the post-normalization dataset
    normalized_dataset_id INT DEFAULT NULL,
    -- the dataset id for the post-format conversion dataset
    converted_dataset_id INT DEFAULT NULL,
    -- the height of the resized images
    resize_height SMALLINT DEFAULT NULL,
    -- the width of the resized images
    resize_width SMALLINT DEFAULT NULL,
    -- the interpolation method used for resizing
    interpolation CHAR(13) DEFAULT NULL,
    -- random seed used for splitting the dataset
    seed INT DEFAULT NULL,
    -- whether or not any intermediate datasets are cleaned up
    cleanup BOOLEAN NOT NULL,
    -- the percentage of the dataset allocated for training
    train_percent SMALLINT DEFAULT NULL,
    -- the percentage of the dataset allocated for validation
    val_percent SMALLINT DEFAULT NULL,
    -- the percentage of the dataset allocated for testing
    test_percent SMALLINT DEFAULT NULL,
    -- the path to the per-pixel normalization statistics
    stats_path VARCHAR(300) DEFAULT NULL,
    -- the normalization method used
    norm_method VARCHAR(32) DEFAULT NULL,
    -- the file format/extension used in the conversion
    conversion_format VARCHAR(5) DEFAULT NULL,
    -- the pillow image quality setting used in jpeg conversion, if applicable
    jpeg_quality SMALLINT DEFAULT NULL,
    -- notes about the processing job
    notes TEXT DEFAULT NULL,
    -- set the processing job id as the primary key
    PRIMARY KEY (id),
    -- set the dataset ids to be foreign keys
    FOREIGN KEY (cleaned_dataset_id) REFERENCES datasets (id),
    FOREIGN KEY (resized_dataset_id) REFERENCES datasets (id),
    FOREIGN KEY (split_dataset_id) REFERENCES datasets (id),
    FOREIGN KEY (normalized_dataset_id) REFERENCES datasets (id),
    FOREIGN KEY (converted_dataset_id) REFERENCES datasets (id),
    -- set the normalization method as a foreign key
    FOREIGN KEY (norm_method) REFERENCES normalization_methods (method),
    -- set the conversion format as a foreign key
    FOREIGN KEY (conversion_format) REFERENCES file_formats (format)
);