#!/usr/bin/env python
# -*- coding: utf-8 -*- 


"""--------------------------------------------------------------------
ZOO
Investigating the Benford's Law
Started on the 29/12/2016

theo.alves.da.costa@gmail.com
https://github.com/theolvs
------------------------------------------------------------------------
"""



import numpy as np
import pandas as pd



def random_cumulative_numbers(n = 10,n_repeat = 100,min = 1,max = 100):
    return [np.sum(np.random.randint(min,max,n)) for x in range(n_repeat)]


def draw_distribution_first_digit(n = 10,n_repeat = 100,min = 1,max = 100):
    distribution = random_cumulative_numbers(n,n_repeat,min,max)
    distribution = list(map(lambda x: int(str(x)[-1]),distribution))
    distribution = pd.Series(distribution)
    distribution = distribution.value_counts()
    distribution.sort_index().plot(kind = "bar")
    return distribution

def draw_distribution_last_digit(n = 10,n_repeat = 100,min = 1,max = 100):
    distribution = random_cumulative_numbers(n,n_repeat,min,max)
    distribution = list(map(lambda x: int(str(x)[0]),distribution))
    distribution = pd.Series(distribution)
    distribution = distribution.value_counts()
    distribution.sort_index().plot(kind = "bar")
    return distribution





