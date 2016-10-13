import numpy as np
import pandas as pd
from sys import argv
import matplotlib.pyplot as plt

data = pd.read_csv(argv[1])
print data

fig, ax = plt.subplots()

plt.plot(data['epoch'], data['acc'], label="training")
plt.plot(data['epoch'], data['val_acc'], label="validation")
plt.xlabel("numper of epochs")
plt.ylabel("accuracy")
plt.legend(loc='lower right')

plt.figure(2)
plt.plot(data['epoch'], data['loss'], label="training")
plt.plot(data['epoch'], data['val_loss'], label="validation")
plt.xlabel("numper of epochs")
plt.ylabel("loss")
plt.legend()

#plt.ylabel('some numbers')


plt.show()
