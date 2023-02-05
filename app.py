from re import S
import pandas as pd
import streamlit as st
import numpy as np

import home
import NB
import RF
import FNN

st.set_page_config(page_title='Machine Learning Demostration', page_icon=None, layout="wide",
                    initial_sidebar_state="auto", menu_items=None)
PAGES = {
    "Naive Bayes" : NB,
    "Random Forest" : RF,
    "Feedforward Neural Network" : FNN
}

st.sidebar.title('Models')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()









