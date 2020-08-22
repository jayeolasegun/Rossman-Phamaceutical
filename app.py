# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 11:52:50 2020

@author:    Jayeola Gbenga

"""
import streamlit as st
import awesome_streamlit as ast


import main.Home
import main.data_overview 
import main.data_visualisation
import main.prediction


ast.core.services.other.set_logging_format()

NAV = {
    
    "Home": main.Home,
    "Sample Data":main.data_overview,
    "Data visualisations": main.data_visualisation,
    "Predictions": main.prediction,
    
}


def main():
    
    """Main function of the App"""
    
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(NAV.keys()))

    page = NAV[selection]

    with st.spinner(f"Loading {selection} ..."):
        ast.shared.components.write_page(page)

    st.sidebar.title("About")
    st.sidebar.info(
        
        """
        This Dashboards enables Rosssman Sales Manager to easily make business decisions 
        concerning sales and Customers
        
        """
    )


if __name__ == "__main__":
    main()

