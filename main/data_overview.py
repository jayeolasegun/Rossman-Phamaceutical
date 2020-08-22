# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 12:01:43 2020

@author: Jayeola Gbenga
"""

import streamlit as st
import pandas as pd 

# import awesome_streamlit as ast


# pylint: disable=line-too-long
def write():
    
    """Used to write the page in the app.py file"""
    
    with st.spinner("Loading Data ..."):
        
        # ast.shared.components.title_awesome("Rossmann Pharmaceuticals 💊🩸🩺🩹💉 ")
        st.title('Data description')
        
        st.write("""
        The  Data contains Sales, Customers, Promo and Holiday information for 1115 Stores 
        oF Rossmann Phamaceuticals 
    
            
        """)
        
        
        
        train = pd.read_csv('data/train.csv')
        store = pd.read_csv('data/store.csv')
        train_df = pd.merge(left = train, right = store, how = 'inner', on = 'Store').set_index('Date')
        
        st.write(train_df.sample(50))

