#!/usr/bin/env python
# -*- coding: utf-8 -*- 

"""--------------------------------------------------------------------
COMPUTER VISION
Algorithms for computer vision
Started on the 28/12/2016

theo.alves.da.costa@gmail.com
https://github.com/theolvs
------------------------------------------------------------------------
"""


from keras.models import Sequential
from keras.optimizers import SGD,RMSprop, Adam
from keras.utils import np_utils
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D,ZeroPadding2D
from keras.callbacks import EarlyStopping
from keras import backend as K


def FCC_model_1():
    model = Sequential()
    model.add(Dense(4000, input_dim=3072))
    model.add(Activation('relu'))
    model.add(Dense(4000))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    return model



def CNN_model_1(input_dim,output_dim):
    model = Sequential()

    model.add(Convolution2D(32, 3,3, border_mode='same',input_shape=input_dim))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3,3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim))
    model.add(Activation('softmax'))

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

    return model






def CNN_model_2(input_dim,output_dim):
    optimizer = RMSprop(lr=1e-4)
    objective = 'categorical_crossentropy'

    def center_normalize(x):
        return (x - K.mean(x)) / K.std(x)

    model = Sequential()

    model.add(Activation(activation=center_normalize, input_shape=input_dim))

    model.add(Convolution2D(32, 5, 5, border_mode='same', activation='relu', dim_ordering='tf'))
    model.add(Convolution2D(32, 5, 5, border_mode='same', activation='relu', dim_ordering='tf'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))

    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))

    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))
    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))

    model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))
    model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))


    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(output_dim))
    model.add(Activation('softmax'))

    model.compile(loss=objective, optimizer=optimizer,metrics=['accuracy'])

    return model


def CNN_model_3(input_dim,output_dim,lr = 1e-2):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=input_dim, dim_ordering='tf'))
    model.add(Convolution2D(4, 3, 3, activation='relu', dim_ordering='tf'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(4, 3, 3, activation='relu', dim_ordering='tf'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='tf'))

    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='tf'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='tf'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='tf'))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim, activation='softmax'))

    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy',metrics=['accuracy'])

    return model


def CNN_model_4(input_dim,output_dim,lr = 1e-2):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=input_dim, dim_ordering='tf'))
    model.add(Convolution2D(32, 3, 3, activation='relu', dim_ordering='tf'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(32, 3, 3, activation='relu', dim_ordering='tf'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='tf'))

    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(64, 3, 3, activation='relu', dim_ordering='tf'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(64, 3, 3, activation='relu', dim_ordering='tf'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='tf'))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim, activation='softmax'))

    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy',metrics=['accuracy'])

    return model







def friends_face_model(input_dim,output_dim):
    model = Sequential()

    model.add(Convolution2D(32, 10,10, border_mode='same',input_shape=input_dim))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 10,10))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 20,20, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 20,20))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))


    model.add(Convolution2D(128, 10,10, border_mode='same'))
    model.add(Activation('relu'))	
    model.add(Convolution2D(128, 10,10))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim))
    model.add(Activation('softmax'))

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

    return model


