#!/usr/bin/env python
# -*- coding: utf-8 -*- 

"""--------------------------------------------------------------------
INSTACART ORDER
Started on the 20/05/2017

theo.alves.da.costa@gmail.com
https://github.com/theolvs
------------------------------------------------------------------------
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

from ai.data.ontologies import Data



#=============================================================================================================================
# PRODUCTS
#=============================================================================================================================




class Products(Data):
    def __init__(self):

        self.load_data()
        self.merge()


    def load_data(self):
        self.data = pd.read_csv("products.csv")
        self.aisles = pd.read_csv("aisles.csv")
        self.departments  = pd.read_csv("departments.csv")
        

    def merge(self):
        self.data = pd.merge(self.data,self.aisles,on = "aisle_id",how = "left")
        self.data = pd.merge(self.data,self.departments,on = "department_id",how = "left")











#=============================================================================================================================
# ORDERS
#=============================================================================================================================


class Orders(Data):
    def __init__(self,checkpoint = True):

        if checkpoint:
            print(">> Reloading pre matched data")
            self.reload_data()

        else:
            print(">> Loading data ...")
            self.load_data()

            print(">> Merging to get product_details ...")
            self.get_product_details()

            print(">> Building product lists")
            self.build_product_list()


    #--------------------------------------------------------------------------------------
    # IO


    def load_data(self):
        self.by_user = pd.read_csv("orders.csv")
        self.by_products_prior = pd.read_csv("order_products__prior.csv")
        self.by_products_train = pd.read_csv("order_products__train.csv")


    def reload_data(self):
        self.by_user = pd.read_pickle("orders_matched.pkl")






    #--------------------------------------------------------------------------------------
    # DATA PREPARATION


    def get_product_details(self):
        products = Products()
        self.by_products_prior = pd.merge(self.by_products_prior,products.data[["product_id","aisle_id","department_id"]],on = "product_id",how = "left")
        self.by_products_train = pd.merge(self.by_products_train,products.data[["product_id","aisle_id","department_id"]],on = "product_id",how = "left")



    def build_product_list(self):

        # AGGREGATION DICTIONARY
        aggregation_dictionary = {
            "product_id": lambda x: " ".join(map(str,x)),
            "aisle_id": lambda x: " ".join(map(str,x)),
            "department_id": lambda x: " ".join(map(str,x)),
            "reordered": lambda x: " ".join(map(str,x)),
            "add_to_cart_order":"count",
        }

        # ON PRIOR AND TRAIN
        grouped_data = self.by_products_prior.groupby("order_id",as_index = False).agg(aggregation_dictionary)
        grouped_data = grouped_data.append(self.by_products_train.groupby("order_id",as_index = False).agg(aggregation_dictionary))
        grouped_data.rename(columns = {"product_id":"products","aisle_id":"aisles","department_id":"departments","add_to_cart_order":"count"},inplace = True)


        # MERGE
        self.by_user = pd.merge(self.by_user,grouped_data,on = "order_id",how = "left")






    #--------------------------------------------------------------------------------------
    # DATA MODELLING

    def build_dataset(self,from_level = "product",to_level = None):
        pass




        
    #--------------------------------------------------------------------------------------
    # NETWORK ANALYSIS


    def build_co_occurences_matrix(self,level = "product",top = None):

        # PROTECTION FOR THE ARGUMENTS
        if level not in ["product","aisle","department"]:
            raise ValueError("Incorrect level entered")

        data = {}

        print(">> Fetching product list")
        products = Products()

        unique_elements = list(products.data[level + "_id"].unique())

        final_data = pd.DataFrame(columns = unique_elements,index = unique_elements).fillna(0)

        # ITERATION OVER EACH ORDER
        n_orders = np.min([len(self.by_user),top])

        for i in range(n_orders):
            print("\r[{}/{}] orders processed".format(i+1,n_orders),end = "")

            order = self.by_user.iloc[i].loc[level + "s"]

            if pd.isnull(order):
                continue

            elements = order.split(" ")

            for i in range(len(elements) - 1):
                element = int(elements[i])
                next_element = int(elements[i + 1])

                final_data.loc[element,next_element] += 1

                # if element not in data:
                #     data[element] = {}

                # if next_element in data[element]:
                #     data[element][next_element] += 1
                # else:
                #     data[element][next_element] = 1

        print("")

        # STORE THE RESULTS IN A DATAFRAME MATRIX
        print(">> Storing ")

        # data = pd.DataFrame(data) #,columns = unique_elements,index = unique_elements)

        # # FILL MISSING VALUES
        # data = data.fillna(0)


        return final_data