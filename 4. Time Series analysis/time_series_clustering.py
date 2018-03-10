#!/usr/bin/env python
# -*- coding: utf-8 -*- 


__author__ = "Theo"



"""--------------------------------------------------------------------
TIME SERIES CLUSTERING
Started on the 2018/03/07

theo.alves.da.costa@gmail.com
https://github.com/theolvs
------------------------------------------------------------------------
"""



# Usual libraries
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import time
from tqdm import tqdm
import requests
import json

# Plotting libraries
import plotly.graph_objs as go
from plotly.offline import iplot,init_notebook_mode
import cufflinks as cf



#=============================================================================================================================
# ALPHA VANTAGE WRAPPER
#=============================================================================================================================



class AlphaVantage(object):
    """Wrapper for https://www.alphavantage.co/ API
    """
    def __init__(self,api_key):
        self.api_key = api_key
        
    def get(self,ticker):
        data = requests.get("https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={}&apikey={}".format(ticker,self.api_key)).json()
        data = pd.DataFrame(data["Time Series (Daily)"]).astype(float).transpose()
        data.columns = ["open","high","low","close","volume"]
        data.index = pd.to_datetime(data.index)
        return data




#=============================================================================================================================
# COMPANY ONTOLOGIES
#=============================================================================================================================



class Company(object):
    def __init__(self,ticker,alpha = None,data = None):
        self.ticker = ticker

        if data is None:
            self.data = alpha.get(ticker)
        else:
            self.data = data
        
    def __repr__(self):
        return self.ticker
    
    def plot(self,variable = "close"):
        if type(variable) != list: variable = [variable]
        fig = self.data[variable].iplot(world_readable=True,asFigure=True)
        iplot(fig)







#=============================================================================================================================
# COMPANY ONTOLOGIES
#=============================================================================================================================




class Companies(object):
    """Companies wrapper
    """
    def __init__(self,tickers = None,companies = None,json_path = None,alpha = None,max_retries = 5):
        """Initialization
        """
        if companies is not None:
            self.data = companies
        elif tickers is not None:
            self.data,_ = self.get_data(tickers,alpha = alpha,max_retries = max_retries)
        elif json_path is not None:
            self.data = self.load_json_data(json_path)



    #--------------------------------------------------------------------------
    # OPERATORS
            
    def __repr__(self):
        return "{} companies in the dataset".format(len(self))
    
    def __len__(self):
        return len(self.data)
    
    def __iter__(self):
        return iter(self.data)
    
    def __getitem__(self,key):
        if type(key) != list : key = [key]
        return [company for company in self if company.ticker in key]
    
    



    #--------------------------------------------------------------------------
    # IO

    def save_as_json(self,json_path):
        data = {}
        for company in self:

            df = company.data.copy()
            df.index = df.index.map(str)
            data[company.ticker] = json.loads(df.to_json())

        with open(json_path, 'w') as file:
            json.dump(data, file,indent = 4,sort_keys = True)


    def load_json_data(self,json_path):
        json_data = json.loads(open(json_path,"r").read())
        
        companies = []

        for ticker,data in json_data.items():
            data = pd.DataFrame(data)
            data.index = pd.to_datetime(data.index)

            company = Company(ticker = ticker,data = data)
            companies.append(company)


        return companies





    #--------------------------------------------------------------------------
    # GETTERS


    def get_data(self,tickers,alpha,max_retries = 5):
        """Financial data getter via API
        """
        data = []
        skipped = []
        for ticker in tqdm(tickers,desc = "Acquiring data"):
            try:
                company = Company(ticker,alpha = alpha)
                data.append(company)
            except Exception as e:
                time.sleep(10)
                skipped.append(ticker)

        i = 0
        while len(skipped) > 0 and i <= max_retries:
            new_data,skipped = self.get_data(skipped,alpha = alpha,max_retries = 0)
            data.extend(new_data)
            i += 1

        return data,skipped



    def get_dataframe(self,tickers = None,variable = "close",normalization = True):
        """Create dataframe from companies data
        """
        if tickers is not None:
            companies = self[tickers]
        else:
            companies = self.data       
        data = pd.concat([company.data[[variable]].rename(columns = {variable:company.ticker}) for company in self],axis = 1)
        
        if normalization:
            data /= data.max(axis = 0)
            
        return data
    


    #--------------------------------------------------------------------------
    # VISUALIZATION
    
    def plot(self,tickers = None,variable = "close"):
        data = self.get_dataframe(tickers,variable)
        fig = data.iplot(world_readable=True,asFigure=True)
        iplot(fig)