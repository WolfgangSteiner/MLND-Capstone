from Training import Training
from CharacterGenerator import CharacterGenerator

t = Training(mean=126.22649, std=60.05183)
t.conv(32)
t.maxpool()
t.conv(64)
t.maxpool()
t.dense(128)
t.dropout(0.5)
t.classifier(10)
t.train_generator()
