from Training import Training
t = Training(input_shape=[32,16,1])
t.use_batchnorm=False
t.winit='normal'
t.lr = 0.00025
t.wreg = 0.0
t.conv(32,3)
t.conv(32,3)
t.maxpool()
t.conv(64,3)
t.conv(64,3)
t.maxpool()
t.conv(128,3)
t.conv(128,3)
t.maxpool()
t.dense(256)
t.dense(256)
t.binary_classifier()
options={'max_blur':2.5, 'max_rotation':15, 'min_size': 0.5, 'min_color_delta': 8, 'min_noise':4, 'max_noise':8}
options['num_epochs'] = 200
t.train_segmentation_generator(options=options)
