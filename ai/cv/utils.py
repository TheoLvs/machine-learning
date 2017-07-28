#!/usr/bin/env python
# -*- coding: utf-8 -*- 

"""--------------------------------------------------------------------
COMPUTER VISION
Utils for computer vision
Started on the 28/12/2016

theo.alves.da.costa@gmail.com
https://github.com/theolvs
------------------------------------------------------------------------
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import os

from sklearn.model_selection import train_test_split



def plot_gallery(images, titles,n_row=3, n_col=6):
    #Helper function to plot a gallery of portraits
    plt.figure(figsize=(2.3 * n_col, 2.3 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i],interpolation="nearest")
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


def plot_image(array,title = ""):
    plt.title(title)
    plt.imshow(array,interpolation='nearest')
    plt.show()


def read_image_from_file(file_path,colors = False,resize = (0,0)):
    img = scipy.misc.imread(file_path,flatten = not colors)
    if resize != (0,0):
        img = scipy.misc.imresize(img,resize,'cubic')

    return img


def build_dataset_from_folder(folder,colors = False,resize = (0,0)):
    X = []
    list_of_images = [folder +"/"+x for x in os.listdir(folder)]
    for i,image in enumerate(list_of_images):
        try:
            print("\rReading from folder %s : [%s/%s]"%(folder,i+1,len(list_of_images)),end = "")
            X.append(read_image_from_file(image,colors = colors,resize = resize))
        except OSError:
            pass
            
    print('')
    return np.array(X)



