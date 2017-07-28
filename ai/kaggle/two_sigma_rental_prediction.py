#!/usr/bin/env python
# -*- coding: utf-8 -*- 

"""--------------------------------------------------------------------
AI UTILS
Utils for various ai and data applications
Started on the 04/03/2016

theo.alves.da.costa@gmail.com
https://github.com/theolvs
------------------------------------------------------------------------
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time





def handling_numerical_variables(data,variable):
    return data[variable]




def handling_dates(data,variable,format = None):
    dates_df = data[[variable]].copy()
    dates_df[variable] = pd.to_datetime(dates_df[variable])
    dates_df = pd.concat([dates_df,pd.DataFrame(list(dates_df[variable].map(extract_feature_from_date)),index = dates_df.index)],axis = 1)
    return dates_df


def extract_feature_from_date(date):
    """Works on a an already datetime"""
    features = {
        "month":date.month,
        "day":date.dayofweek,
        "date":str(date.date()),
        "time":str(date.time())[:5],
        "year":date.year,
        "hour":date.hour,
        "moment_of_day":extract_moment_of_day_from_date(date)
    }
    return features



def extract_moment_of_day_from_date(date):
    """Split the day in 6 moments"""
    return int(date.hour/6)



















