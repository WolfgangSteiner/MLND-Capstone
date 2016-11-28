from Training import Training
t = Training(input_shape=[32,16,1])
t.conv(32,5)
t.conv(32,5)
t.maxpool()
t.dense(256)
t.dense(256)
t.dropout(0.5)
t.binary_classifier()
t.train_segmentation_generator(options={'max_blur':2.5, 'max_rotation':5})