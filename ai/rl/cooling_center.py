#!/usr/bin/env python
# -*- coding: utf-8 -*- 


"""--------------------------------------------------------------------
REINFORCEMENT LEARNING
COOLING CENTER

Started on the 25/08/2017


theo.alves.da.costa@gmail.com
https://github.com/theolvs
------------------------------------------------------------------------
"""


import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import sys
import random
import time
from tqdm import tqdm
from collections import Counter

from scipy import stats

# Deep Learning (Keras, Tensorflow)
import tensorflow as tf
from keras.models import Sequential
from keras.optimizers import SGD,RMSprop, Adam
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import MaxPooling2D,ZeroPadding2D,Conv2D
from keras.utils.np_utils import to_categorical






#===========================================================================================================
# COOLING CENTER ENVIRONMENT
#===========================================================================================================



class CoolingCenter(object):
    def __init__(self,levels_activity = 20,levels_cooling = 10,gamma = 5):

        self.hour = 0
        self.gamma = gamma
        self.levels_activity = levels_activity
        self.levels_cooling = levels_cooling
        self.define_activity(levels_activity)
        self.define_cooling(levels_cooling)

        
    def define_activity(self,levels_activity):
        # Define the peaks of activity
        peak_morning = np.random.randint(7,10)
        peak_evening = np.random.randint(17,22)

        # Build the distribution
        x1 = np.array(stats.poisson.pmf(range(24),peak_morning))
        x2 = np.array(stats.poisson.pmf(range(24),peak_evening))
        x = x1 + x2
        x *= (100/0.14)

        # Discretize the distribution
        take_closest = lambda j,vector:min(vector,key=lambda x:abs(x-j))
        percentiles = np.percentile(x,range(0,100,int(100/levels_activity)))
        assert len(percentiles) == levels_activity
        x_disc = np.array([take_closest(y,percentiles) for y in x])

        # Store the variable
        self.observation_space = percentiles
        self.activity = np.expand_dims(x_disc,axis = 0)



    def define_cooling(self,levels_cooling):
        self.action_space = range(0,100,int(100/levels_cooling))
        assert len(self.action_space) == levels_cooling

        initial_value = random.choice(self.action_space)
        self.cooling = np.full((1,24),initial_value)



    def reset(self):
        self.__init__(self.levels_activity,self.levels_cooling,self.gamma)
        return self.reset_state()

    def reset_state(self):
        activity = self.activity[0][0]
        activity_state = self.convert_activity_to_state(activity)
        return activity_state


    def convert_activity_to_state(self,activity):
        state = int(np.where(self.observation_space == activity)[0][0])
        return state



    def render(self):
        # Show the activity and cooling
        plt.figure(figsize = (14,5))
        plt.plot(np.squeeze(self.activity),c = "red",label = "activity")
        plt.plot(np.squeeze(self.cooling),c = "blue",label = "cooling")
        plt.legend()
        plt.show()

        # Show the rewards
        plt.figure(figsize = (14,5))
        rewards,winnings,losses = self.compute_daily_rewards()
        plt.title("Total reward : {}".format(int(np.sum(rewards))))
        plt.plot(rewards,c = "blue",label = "rewards")
        plt.plot(losses*(-1),c = "red",label = "losses")
        plt.plot(winnings,c = "green",label = "winnings")
        plt.legend()
        plt.show()


    def compute_reward(self,activity,cooling):

        # CALCULATING THE WINNINGS
        win = activity

        # CALCULATING THE LOSSES
        if cooling >= activity:
            loss = self.gamma * (cooling - activity)
        else:
            difference = (activity-cooling)/(cooling+1)
            default_probability = np.tanh(difference)
            if np.random.rand() > default_probability:
                loss = 0
            else:
                loss = np.random.normal(loc = 1.6,scale = 0.4) * 100

        return win,loss



    def compute_daily_rewards(self):
        winnings = []
        losses = []
        rewards = []
        for i in range(24):
            activity = self.activity[0][i]
            cooling = self.cooling[0][i]
            win,loss = self.compute_reward(activity,cooling)
            winnings.append(win)
            losses.append(loss)
            rewards.append(win-loss)

        return np.array(rewards),np.array(winnings),np.array(losses)



    def step(self,cooling_action):

        # Convert cooling_action to cooling_value
        cooling = self.action_space[cooling_action]

        # Update the cooling
        self.cooling[0][self.hour] = cooling

        activity = self.activity[0][self.hour]
        win,loss = self.compute_reward(activity,cooling)
        reward = win-loss

        self.hour += 1

        if int(self.hour) == 24:
            new_state = self.reset_state()
            done = True
        else:
            new_activity = self.activity[0][self.hour]
            new_state = self.convert_activity_to_state(new_activity)
            done = False


        return new_state,reward,done


    
    


