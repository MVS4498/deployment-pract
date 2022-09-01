import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
#Category Imputer
from feature_engine.imputation import CategoricalImputer
#Encoding
from sklearn.preprocessing import OrdinalEncoder
#Numerical Imputer
from sklearn.impute import KNNImputer
#Scaling
from sklearn.preprocessing import StandardScaler
#Model
from sklearn.tree import DecisionTreeRegressor
#Importing Pipeline
import pickle

pipe = pickle.load(open('LinearModelPipe.pkl' , 'rb'))

################## Interface building ##################################
st.title('Give the unnamed: 0 col value.')

Unnamed = st.number_input('Unnamed: 0',min_value=0,max_value=100)
u = st.slider('U', min_value=0 , max_value=1000)

input_data = {
    'Unnamed: 0':Unnamed, 
    'u': u
}
input_data = pd.DataFrame([input_data])
st.write(input_data)
prediction = pipe.predict(input_data)

if st.button('Predict'):
    st.success(prediction[0])
