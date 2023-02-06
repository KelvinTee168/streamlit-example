from re import S
import pandas as pd
import streamlit as st
import numpy as np
import pickle
import random


#@st.cache
def get_dict():
    print("Loading Dict")
    return pickle.load((open('categorical_dict.pkl', 'rb')))

#@st.cache
def load_model():
    print("Loading model")
    return pickle.load(open('nb_3.pkl', 'rb'))


def app():
    st.title("Naive Bayes Prediction")
    
    mode = st.radio("What\'s your prediction mode",('custom data', 'csv file'))
    df = pd.DataFrame() 
    loaded_model = load_model()
    categorical_dict = get_dict()
    
    if mode == 'csv file':
        uploaded_file = st.file_uploader("Choose a file for prediction", type=['csv','xlsx'],accept_multiple_files=False,key="fileUploader")
        if uploaded_file is not None:
            #read csv
            df=pd.read_csv(uploaded_file)
        else:
            st.warning("you need to upload a csv or excel file.")
    
    
    else:
        st.write("Please enter your own values in the fields and see the predictions below.")
        data = [{
            "USMER":"1",
            "MEDICAL_UNIT":"1",
            "SEX":"Male",
            "HOSPITALIZED":"NO",
            "INTUBED":"NO",
            "PNEUMONIA":"NO",
            "AGE":"20",
            "PREGNANT":"NO",
            "DIABETES":"NO",
            "COPD":"NO",
            "ASTHMA":"NO",
            "INMSUPR":"NO",
            "HYPERTENSION":"NO",
            "OTHER_DISEASE":"NO",
            "CARDIOVASCULAR":"NO",
            "OBESITY":"NO",
            "RENAL_CHRONIC":"NO",
            "TOBACCO":"NO",
            "ICU":"NO",
            "DEATH":"NO",
        }]


        for key in data[0]:
            if(key != "USMER" and key != "MEDICAL_UNIT" and key != "AGE" and key != "CLASIFFICATION_FINAL"):
                data[0][key] = st.selectbox(key,list(categorical_dict[key]))
            elif(key == "USMER"):
                data[0][key] = st.number_input(key, min_value=1, max_value=2, value=1, step=1)
            elif(key == "MEDICAL_UNIT"):
                data[0][key] = st.number_input(key, min_value=1, max_value=13, value=1, step=1)
            elif(key == "AGE"):
                data[0][key] = st.number_input(key, min_value=1, max_value=121, value=20, step=1)

        df = pd.DataFrame.from_records(data=data)
    
    if not df.empty:
        st.dataframe(df)
        rfe_top20 = pd.read_csv('RFE_Top20.csv')

        col_list = [col for col in df.columns.tolist() if df[col].dtype.name == "object"]
        df_oh = df[col_list]
        df = df.drop(col_list, 1)
        df_oh = pd.get_dummies(df_oh)
        df = pd.concat([df, df_oh], axis=1)

        X = df.copy()

        for feature in rfe_top20['Features']:
            if feature not in df.columns:
                X[feature] = 0

        pred = loaded_model.predict(X[rfe_top20['Features']])
        st.write("## Prediction")
        if pred.shape[0] > 1:
            st.write('Predicted CLASSIFICATION_FINAL: ', pred)
        else:
            st.write('Predicted CLASSIFICATION_FINAL: ', pred[0])

