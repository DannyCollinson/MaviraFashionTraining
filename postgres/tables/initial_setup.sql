/*
 Insert some initial data into some tables
 */
-- Insert some default normalization methods
INSERT INTO normalization_methods (method, notes) VALUES
    ('minmax', 'Min-max scaling. (val - min) / (max - min)'),
    ('zscore', 'Per-channel Z-score normalization. (val - mean) / std'),
    ('pixelz', 'Pixel-wise Z-score normalization. (val - mean) / std'),
    (
        'minmaxplus',
        'Min-max scaling to [-1, 1]. 2 * ((val - min) / (max - min)) - 1'
    )
    ('zunknown', 'Either per-channel or pixel-wise Z-score normalization')
    ('unknown', 'Normalization method unknown'),
    ('none', 'No normalization applied');

-- Insert some default conversion formats
INSERT INTO valid_file_formats (format, load_func, notes) VALUES
    ('jpg', 'torchvision.io.decode_image', 'JPEG'),
    ('jpeg', 'torchvision.io.decode_image', 'JPEG'),
    ('pt', 'torch.load', 'PyTorch'),
    ('pth', 'torch.load', 'PyTorch'),
    ('npy', 'np.load', 'NumPy'),
    ('np', 'np.load', 'NumPy');