from Training import Training
t = Training(input_shape=[32,32,1])
t.use_batchnorm = False
t.wreg=0.0
t.lr = 0.001
t.conv(16,3)
t.maxpool()
t.conv(32,3)
t.maxpool()
t.dense(128)
t.dense(128)
t.binary_classifier()
options={'min_color_delta':8, 'min_blur':0.5, 'max_blur':2.5, 'max_rotation':15.0, 'min_noise':4, 'max_noise':8}
options['num_epochs'] = 200
t.train_detection_generator(options=options)
