#!/usr/bin/env python
# -*- coding: utf-8 -*- 

"""--------------------------------------------------------------------
COMPUTER VISION
Scripts for the CIFAR 10 dataset
Started on the 28/12/2016

theo.alves.da.costa@gmail.com
https://github.com/theolvs
------------------------------------------------------------------------
"""

import matplotlib.pyplot as plt


def unpickle(file):
    '''Function to unpickle a file'''
    import pickle
    fo = open(file, 'rb')
    u = pickle._Unpickler(fo)
    u.encoding = 'latin1'
    dict = u.load()
    fo.close()
    return dict

    
def unload_data():    
    for i in range(1,6):
        file_data = unpickle(folder+"data_batch_{0}".format(i))
        data = np.append(data,file_data['data'],axis = 0) if i > 1 else file_data['data']
        labels = np.append(labels,file_data['labels'],axis = 0) if i > 1 else file_data['labels']

    return data,labels



def plot_img(array,title):
    img = array.reshape(3,32,32).transpose(1,2,0)
    plt.title(title)
    plt.imshow(img,interpolation = "nearest")
    plt.show()



def plot_gallery(images, titles,n_row=3, n_col=6):
    print('hello')
    #Helper function to plot a gallery of portraits
    plt.figure(figsize=(2.3 * n_col, 2.3 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape(3,32,32).transpose(1,2,0),interpolation="nearest")
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())



labels = {
    0:"airplane",
    1:"automobile",
    2:"bird",
    3:"car",
    4:"deer",
    5:"dog",
    6:"frog",
    7:"horse",
    8:"ship",
    9:"truck"
}






