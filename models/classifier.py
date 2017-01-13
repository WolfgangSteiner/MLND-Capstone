from Training import Training
t = Training(batch_size=128)
t.lr = 0.00025
t.use_batchnorm = False
t.wreg = 0.0
t.winit='normal'
t.conv(32,3)
t.conv(32,3)
t.maxpool()
t.conv(64,3)
t.conv(64,3)
t.maxpool()
t.conv(128,3)
t.conv(128,3)
t.maxpool()
t.dense(1024)
#t.dropout(0.5)
t.dense(1024)
#t.dropout(0.5)
t.classifier(10)
options={'min_color_delta':8, 'min_blur':0.5, 'max_blur':2.5, 'max_rotation':15.0, 'min_noise':4, 'min_size':0.5, 'max_size':1.0, 'max_noise':8, 'full_alphabet':False}
t.train_generator(options=options)
