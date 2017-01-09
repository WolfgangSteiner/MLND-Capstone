from Training import Training
t = Training(input_shape=[32,32,1])
t.use_batchnorm = False
t.batch_size = 32
t.conv(32,16,subsample=(8,8))
t.dense(32 * 32)
t.sigmoid()
t.train_detection_generator_new(options={'max_blur':2.5, 'max_rotation':5})
