# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 12:01:42 2020

@author: Jayeola Gbenga

"""


import streamlit as st
import awesome_streamlit as ast


# pylint: disable=line-too-long
def write():
    
    """Used to write the page in the app.py file"""
    with st.spinner("Loading Home ..."):
        # ast.shared.components.title_awesome("Rossmann Pharmaceuticals 💊🩸🩺🩹💉 ")
        st.title('Rossmann Phamaceutical Sales Manager Dashboard')
        
        st.write(
            """
            Rossman pharmaceuticals is an international pharamaceutical company with millions of stores acros the globe.
            In Kenya, it has around 1115 stores in major cities and towns. 

            The company is guided by the following virtues:
            - **Practical Wisdom.**
            - **Moral Rule**,  **Moral Virtue** and **Moral Sense**.
            - **Personal Virtue**.
                """
        )
        # ast.shared.components.video_youtube(
        #     src="https://www.youtube.com/embed/B2iAodr0fOo"
        # )
            
            
        #st.image('ross.jpg', use_column_width=True)
        st.write("""
        Like any other business entity, profits have to be made in order to send all employees home happy.

        This web app forecasts the sales across the stores in Kenya for the coming six weeks using past data.""")