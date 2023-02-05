from re import S
import pandas as pd
import streamlit as st
import numpy as np
import pickle
import random
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler


#@st.cache
def get_dict():
    print("Loading Dict")
    return pickle.load((open('categorical_dict.pkl', 'rb')))


def app():
    st.title("Feedforward Neural Network Prediction")
    st.write("Please enter your own values in the fields and see the predictions below.")
    loaded_model = load_model('model_2.h5')
    categorical_dict = get_dict()
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
        else:
            data[0][key] = st.number_input(key, min_value=1, max_value=10, value=5, step=1)
            
    df = pd.DataFrame.from_records(data=data)
    
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
    
    sc = StandardScaler()
    X_ss = sc.fit_transform(X[rfe_top20['Features']])
    
    pred = loaded_model.predict(X_ss)
    
    st.write(pred)
    pred = np.argmax(pred, axis=1)
    
    st.write("## Prediction")
    st.write('Predicted CLASSIFICATION_FINAL: ', pred[0])
