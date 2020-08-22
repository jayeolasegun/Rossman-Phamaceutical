# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 12:01:43 2020

@author: Jayeola Gbenga

"""


import streamlit as st
import awesome_streamlit as ast
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# statistics
from statsmodels.distributions.empirical_distribution import ECDF

# time series analysis
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def write():
    """Used to write the page in the app.py file"""
    with st.spinner("Loading Plots ..."):
        st.title('Data Visualisation ')


        train = pd.read_csv("data/train.csv", parse_dates = True, index_col = 'Date')
        store = pd.read_csv('data/store.csv')
        
        
        
        
        # data extraction
        train['Year'] = train.index.year
        train['Month'] = train.index.month
        train['Day'] = train.index.day
        train['WeekOfYear'] = train.index.weekofyear

        # adding new variable
        train['SalePerCustomer'] = train['Sales']/train['Customers']
        
        train = pd.merge(train, store, how = 'inner', on = 'Store')
    
        #train_df['StateHoliday'] = train_df['StateHoliday'].replace({'0':'None',0 :'None'}) 
            
        #train_df = train_df.drop(train_df[(train_df['Sales'] == 0) & (train_df['Open'] == 0)].index)
            
        
        
        
        
        
        st.sidebar.title("Gallery")
        
        st.sidebar.subheader("Choose Feature or Aspect to plot")
        
        plot = st.sidebar.selectbox("Features", ("Sales Trend","Customers Trend", "Sales per Customer" ))


        if plot == 'Sales Trend':
            
            # if st.sidebar.button("Predict", key='predict'):
            st.subheader("Monthly Sales Trend")
            sns.factorplot(data = train, x = 'Month', y = "Sales", 
               col = 'StoreType', # per store type in cols
               palette = 'plasma',
               hue = 'StoreType',
               row = 'Promo', # per promo in the store in rows
              ) 
            
            plt.grid()
            st.pyplot()
            
            
        if plot == 'Customers Trend':
            st.subheader("Monthly Customers Trend")
            sns.factorplot(data = train, x = 'Month', y = "Customers", 
               col = 'StoreType', # per store type in cols
               palette = 'plasma',
               hue = 'StoreType',
               row = 'Promo', # per promo in the store in rows
               )
            st.pyplot()
            


        if plot == 'Sales per Customer':
            st.subheader("Monthly Sales per customer Trend")
            # sale per customer trends
            sns.factorplot(data = train, x = 'Month', y = "SalePerCustomer", 
               col = 'StoreType', # per store type in cols
               palette = 'plasma',
               hue = 'StoreType',
               row = 'Promo', # per promo in the store in rows
               ) 
            st.pyplot()
            
