
#!/usr/bin/env python
# -*- coding: utf-8 -*- 



#To add : 
#- TextSeq class
# changer vectorization en entier et non booléens


#-------------------------------------------------------------------------------------------------------------------------------------------



# NOTES FROM F CHOLLET OF THE ORIGINAL SCRIPT
# https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
'''Example script to generate text from Nietzsche's writings.
At least 20 epochs are required before the generated text
starts sounding coherent.
It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.
If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''



#-------------------------------------------------------------------------------------------------------------------------------------------



from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras.models import load_model
import numpy as np
import random
import sys
import time



class TextChar():
    def __init__(self,text = "",file = "",encoding = "utf8"):
        
        self.encoding = encoding

        if text == "":
            self.file = file
            self.text = open(self.file,"rb").read().decode(self.encoding).lower()
        else:
            self.text = text.lower()

        print('>> corpus length:', len(self.text))

        self.chars = sorted(list(set(self.text)))
        print('>> total chars:', len(self.chars))

        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))


    def __repr__(self):
        return self.text[:100]

    def __str__(self):
        return self.text[:100]




class RNN_LSTM_Generator():
    def __init__(self,text = "",file = "",maxlen = 40,step = 3,model = ""):
        print("-"*75)
        print("Initialization of a generator model for text source :\n... %s\n"%file)

        self.text = TextChar(text = text,file = file)
        self.maxlen = maxlen

        self.sentences = []
        self.next_chars = []

        for i in range(0, len(self.text.text) - self.maxlen, step):
            self.sentences.append(self.text.text[i: i + self.maxlen])
            self.next_chars.append(self.text.text[i + self.maxlen])
        print('>> nb sequences of len %s:'%self.maxlen, len(self.sentences))



        print('\n>> Vectorization ...')
        self.X = np.zeros((len(self.sentences), self.maxlen, len(self.text.chars)), dtype=np.bool)
        self.y = np.zeros((len(self.sentences), len(self.text.chars)), dtype=np.bool)
        for i, sentence in enumerate(self.sentences):
            for t, char in enumerate(sentence):
                self.X[i, t, self.text.char_indices[char]] = 1
            self.y[i, self.text.char_indices[self.next_chars[i]]] = 1
        print('... Vectorization OK')


        print("\n"+"-"*75)

        if model == "":
            self.build_model()
        else:
            self.load_model(model)


    def load_model(self,model):
        print('>> Retrieving model %s'%model)
        self.model = load_model(model)
        print("... Model correctly retrieved")

    def save_model(self,model):
        self.model.save(model+".h5")



    def build_model(self):
        # build the model: a single LSTM
        print('>> Building model ...')
        self.model = Sequential()
        self.model.add(LSTM(128, input_shape=(self.maxlen, len(self.text.chars))))
        self.model.add(Dense(len(self.text.chars)))
        self.model.add(Activation('softmax'))

        optimizer = RMSprop(lr=0.01)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        print("... Model correctly build")




    def sample(self,preds, temperature=1.0):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)


    def simulation(self,n_iterations = 10,sentence_length = 400,n_epochs = 1):
        self.build_model()
        # train the model, output generated text after each iteration
        for iteration in range(1, n_iterations):
            print()
            print('-' * 50)
            print('Iteration', iteration)
            self.model.fit(self.X, self.y, batch_size=128, nb_epoch=n_epochs)

            diversities = [0.2, 0.5, 1.0, 1.2]
            self.generate_random(diversities,sentence_length)


    def generate_random(self,diversity = 0.4,sentence_length = 400):
        if type(diversity) != list:
            diversity = [diversity]
        start_index = random.randint(0, len(self.text.text) - self.maxlen - 1)

        for d in diversity:
            sentence = self.text.text[start_index: start_index + self.maxlen]
            # generated += sentence
            print('----- Generating with seed: "' + sentence + '"')
            # sys.stdout.write(sentence)

            generated = self.predict(sentence,sentence_length,d)
            print(sentence+generated)
            print()

        return generated



    def train(self,batch_size = 128,n_epochs = 1):
        self.model.fit(self.X,self.y,batch_size = batch_size,nb_epoch = n_epochs)



    def predict_next_char(self,sentence,diversity = 0.4):
        x = self.convert_sentence_to_indices(sentence,self.maxlen)

        preds = self.model.predict(x, verbose=0)[0]
        next_index = self.sample(preds, diversity)
        next_char = self.text.indices_char[next_index]

        return next_char


    def predict(self,sentence,sentence_length,diversity = 0.4):
        full_sentence = sentence
        generated = ""
        for i in range(sentence_length):

            next_char = self.predict_next_char(full_sentence,diversity)
            generated += next_char
            full_sentence = full_sentence[1:] + next_char

            # sys.stdout.write(next_char)
            # sys.stdout.flush()
            # time.sleep(0.2)


        return generated



    def convert_sentence_to_indices(self,sentence,sentence_length = 0):
        if sentence_length == 0:
            sentence_length = len(sentence)

        x = np.zeros((1, sentence_length, len(self.text.chars)))
        for t, char in enumerate(sentence[-self.maxlen:]):
            x[0, t, self.text.char_indices[char]] = 1.
        return x

