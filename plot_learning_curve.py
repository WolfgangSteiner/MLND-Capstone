import numpy as np
import pandas as pd
from sys import argv
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

data=[]
names=[]

for f in argv[1:]:
    names.append(f)
    data.append(pd.read_csv(f+".log"))

def has_lr(d):
    return 'lr' in d.columns

def show_lr():
    for d in data:
        if not has_lr(d):
            return False

    return True

numplots = 3 if show_lr() else 2
plt.rcParams.update({'font.size': 8})
figure, ax = plt.subplots(nrows=numplots)
figure.set_size_inches((7,4.8))
#plt.subplot(111)
data=[]
for f in names:
    data.append(pd.read_csv(f+".log"))

for i in range(0,len(names)):
    ax[0].plot(data[i]['acc'], label="training")
    ax[0].plot(data[i]['val_acc'], label="validation")
    ax[0].set_ylabel("accuracy")

    ax[0].xaxis.set_ticklabels([])
    ax[1].plot(data[i]['loss'], label="training")
    ax[1].plot(data[i]['val_loss'], label="validation")
    ax[1].xaxis.set_ticklabels([])
    ax[1].set_ylabel("loss")

    if show_lr():
        ax[2].plot(data[i]['lr'])
        ax[2].set_yscale('log')
        ax[2].set_ylabel("learning rate")



ax[0].legend(loc="lower right")
ax[1].legend()
plt.xlabel("number of epochs")
plt.savefig("latex/fig/learning_curve.pdf")
