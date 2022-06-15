import streamlit as st
from joblib import load
import pandas as pd


st.write("""
         # Penguin Species Prediction Apps
         
         The model to predict penguin species according to set of attributes given<br>
         """)


st.sidebar.header('User Input Parameter')

st.sidebar.markdown("""
                    [Example of CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
                    """)

# Input user (1)CSV file or (2)Direct Input
loaded_file = st.sidebar.file_uploader("Upload CSV file here", type=['csv'])


if loaded_file is not None:
    data_input = pd.read_csv(loaded_file) ###
    
else:
    def user_input_feature():
        island = st.sidebar.selectbox('island', ('Torgersen', 'Biscoe', 'Dream'))
        sex = st.sidebar.selectbox('sex', ('male', 'female'))
        bill_length_mm = st.sidebar.slider('Bill length (mm)', 32.1,59.6,43.9)
        bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1,21.5,17.2)
        flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 172.0,231.0,201.0)
        body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0,6300.0,4207.0)
        
        input_dict = {
            'island': island, 
            'bill_length_mm': bill_length_mm, 
            'bill_depth_mm': bill_depth_mm,
            'flipper_length_mm': flipper_length_mm, 
            'body_mass_g': body_mass_g, 
            'sex': sex
        }
        
        user_input = pd.DataFrame(input_dict, index=[0])
        return user_input
    
    data_input = user_input_feature()


# concat inputs from user with the df
df = pd.read_csv('penguins_cleaned.csv')
df = pd.concat([data_input, df.drop(['species'], axis=1)], axis=0)

# split categorical & numerical
df_cat = df.select_dtypes(include='object')
df_num = df.select_dtypes(exclude='object')

# get_dummies of categorical
df_cat = pd.get_dummies(df_cat)
df_cat.drop(['island_Biscoe', 'sex_female'], axis=1, inplace=True)

# concat categorical & numerical after encoding
df_dummies = pd.concat([df_cat, df_num], axis=1)
data_test = df_dummies[:1]


loaded_model = load('penguins_model.joblib')

# Display inputs from user
st.write('### User Input Features')
if loaded_file is not None:
    st.write(data_test)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(data_test)


# Prediction
st.write('### Prediction')
y_hat = loaded_model.predict(data_test)
st.write('Result    :',y_hat[0])

# Probability
st.write('### Prediction Probability')
proba = loaded_model.predict_proba(data_test)
proba = pd.DataFrame(proba, columns=['Adelie','Chinstrap','Gentoo'])
st.write(proba)












