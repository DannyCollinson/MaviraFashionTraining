/*
 Define table to store information about datasets
 */
-- If needed, drop table to start
-- leave this commented out unless you know what you are doing
-- DROP TABLE IF EXISTS datasets;
CREATE TABLE datasets (
    -- unique serial number to identify the dataset
    id BIGINT GENERATED ALWAYS AS IDENTITY,
    -- the name of the directory that contains the dataset
    -- relative to the project directory
    dir VARCHAR(300) NOT NULL,
    -- filename extension of the images in the dataset
    extension VARCHAR(5) NOT NULL,
    -- time that the dataset was finished being created
    created TIMESTAMP WITH TIME ZONE DEFAULT NULL,
    -- the number of classes in the dataset
    num_classes SMALLINT NOT NULL,
    -- the total number of images in the dataset
    num_images INT NOT NULL,
    -- boolean for whether or not the filenames in the dataset has been cleaned
    is_cleaned BOOLEAN NOT NULL,
    -- boolean for whether or not the dataset has been resized
    is_resized BOOLEAN NOT NULL,
    -- boolean for whether or not the dataset is split into train/val/test
    is_split BOOLEAN NOT NULL,
    -- boolean for whether or not the dataset has been normalized
    is_normalized BOOLEAN NOT NULL,
    -- boolean saying if the dataset has been converted to another format
    is_converted BOOLEAN NOT NULL,
    -- number of training images in the dataset
    num_train INT DEFAULT NULL,
    -- number of validation images in the dataset
    num_val INT DEFAULT NULL,
    -- number of testing images in the dataset
    num_test INT DEFAULT NULL,
    -- dimension 1 (height) of the images in the dataset
    image_height INT DEFAULT NULL,
    -- dimension 2 (width) of the images in the dataset
    image_width INT DEFAULT NULL,
    -- normalization method used
    norm_method VARCHAR(64) DEFAULT NULL,
    -- notes about the dataset
    notes TEXT DEFAULT NULL,
    -- set the dataset id as the primary key
    PRIMARY KEY (id),
    -- set the filename extension as a foreign key
    FOREIGN KEY (extension) REFERENCES file_formats (format),
    -- set the normalization method as a foreign key
    FOREIGN KEY (norm_method) REFERENCES normalization_methods (method),
    -- add a unique constraint accross all other columns
    -- to prevent double registration of the same dataset
    UNIQUE (
        dir,
        extension,
        num_classes,
        num_images,
        is_cleaned,
        is_resized,
        is_split,
        is_normalized,
        is_converted,
        num_train,
        num_val,
        num_test,
        image_height,
        image_width,
        norm_method
    )
);