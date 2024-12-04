/*
 Insert some initial data into some tables
 */
-- Insert some default normalization methods
INSERT INTO normalization_methods (method, notes) VALUES
    ('zscore', 'Per-channel Z-score normalization. (val - mean) / std'),
    ('pixelz', 'Pixel-wise Z-score normalization. (val - mean) / std'),
    ('localz', 'Z-score normalization per image'),
    ('minmax', 'Min-max scaling. (val - min) / (max - min)'),
    (
        'minmaxextended',
        'Min-max scaling to [-1, 1]. 2 * ((val - min) / (max - min)) - 1'
    ),
    ('localminmax', 'Min-max scaling per image'),
    ('localminmax_extended', 'Min-max scaling to [-1, 1] per image'),
    (
        'z_unknown',
        'Either per-channel, pixel-wise, or local Z-score normalization'
    ),
    ('minmax_unknown', 'Min-max scaling to [0, 1] globally or per image'),
    ('unknown', 'Normalization method unknown'),
    ('none', 'No normalization')
;

-- Insert some default conversion formats
INSERT INTO file_formats (format, load_func, save_func, notes) VALUES
    ('jpg', 'torchvision.io.decode_image', 'Image.save', 'JPEG'),
    ('jpeg', 'torchvision.io.decode_image', 'Image.save', 'JPEG'),
    ('png', 'torchvision.io.decode_image', 'Image.save', 'PNG'),
    ('pt', 'torch.load', 'torch.save', 'PyTorch'),
    ('pth', 'torch.load', 'torch.save', 'PyTorch'),
    ('npy', 'np.load', 'np.save', 'NumPy'),
    ('np', 'np.load', 'np.save', 'NumPy'),
    ('npz', 'np.load', 'np.save', 'NumPy')
;