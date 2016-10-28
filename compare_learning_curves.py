import numpy as np
import pandas as pd
from sys import argv
import matplotlib.pyplot as plt
from time import sleep

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

figure, ax = plt.subplots(numplots)
#plt.subplot(111)
plt.ion()


while True:
    ax[0].clear()
    ax[1].clear()

    data=[]
    for f in names:
        data.append(pd.read_csv(f+".log"))

    for i in range(0,len(names)):
        ax[0].plot(data[i]['epoch'], data[i]['acc'], label= names[i] + " (tr)")
        ax[0].plot(data[i]['epoch'], data[i]['val_acc'], label=names[i] + " (val)")
        ax[1].plot(data[i]['epoch'], data[i]['loss'], label= names[i] + " (tr)")
        ax[1].plot(data[i]['epoch'], data[i]['val_loss'], label=names[i] + " (val)")

        if show_lr():
            ax[2].plot(data[i]['epoch'], data[i]['lr'])
            ax[2].set_yscale('log')
            

    ax[0].legend(loc="lower right")
    ax[1].legend()
    plt.xlabel("number of epochs")
    plt.draw()
    plt.pause(1)

