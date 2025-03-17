import streamlit as st
import pandas as pd
import numpy as np
import time
import re
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from streamlit_ace import st_ace

def contains_symbols(df,column):
    return df[column].astype(str).str.contains(r'[^\d.]').any()


def change_data_types(df, column, convert_to):
                if convert_to == 'int':
                    if contains_symbols(df, column):
                        df[column] = df[column].apply(lambda x : float(re.sub(r'[^\d.]', '' , x)))
                    df[column] = df[column].astype(int)
                    st.success(f'✅ Successfully Changed Data Type of {column} to int')
                    st.session_state.to_clean = df  
                    st.session_state.history_tracker.save_version(st.session_state.to_clean, action=f"Changed Data Type of {column} to {convert_to}")
                    st.info("Clean data has been stored for other processes. Select what else you'll like to do on the side bar")   

                elif convert_to == 'float':
                    if contains_symbols(df, column):
                        df[column] = df[column].apply(lambda x : float(re.sub(r'[^\d.]', '' , x)))
                    df[column] = df[column].astype(float)
                    st.success(f'✅ Successfully Changed Data Type of {column} to float')
                    st.session_state.to_clean = df  
                    st.session_state.history_tracker.save_version(st.session_state.to_clean, action=f"Changed Data Type of {column} to {convert_to}")
                    st.info("Clean data has been stored for other processes. Select what else you'll like to do on the side bar")   

                elif convert_to == 'object':   
                    df[column] = df[column].astype(str)
                    st.success(f'✅ Successfully Changed Data Type of {column} to object')
                    st.session_state.to_clean = df  
                    st.session_state.history_tracker.save_version(st.session_state.to_clean, action=f"Changed Data Type of {column} to {convert_to}")
                    st.info("Clean data has been stored for other processes. Select what else you'll like to do on the side bar") 

                elif convert_to == 'datetime': 
                     df[column] = pd.to_datetime(df[column])
                     st.success(f'✅ Successfully Changed Data Type of {column} to datetime')
                     st.session_state.to_clean = df 
                     st.session_state.history_tracker.save_version(st.session_state.to_clean, action=f"Changed Data Type of {column} to {convert_to}") 
                     st.info("Clean data has been stored for other processes. Select what else you'll like to do on the side bar")  


def binning(df, column):
        num_bins = st.number_input('Enter the number of bins:', min_value=2, step=1, value=5)
        binning_method = st.selectbox('Select binning method:', ['Equal Width', 'Equal Frequency'], index=None)

        if st.button('Apply Binning'):
            if binning_method == 'Equal Width':
                df[column + '_binned'] = pd.cut(df[column], bins=num_bins, labels=False)
                st.session_state.to_clean = df  
                st.session_state.history_tracker.save_version(st.session_state.to_clean, action=f"Binned {column} with {binning_method}")
                st.info("Clean data has been stored for other processes. Select what else you'll like to do on the side bar")   
            
            elif binning_method == 'Equal Frequency':
                df[column + '_binned'] = pd.qcut(df[column], q=num_bins, labels=False, duplicates='drop')
                st.session_state.to_clean = df  
                st.session_state.history_tracker.save_version(st.session_state.to_clean, action=f"Binned {column} with {binning_method}")
                st.info("Clean data has been stored for other processes. Select what else you'll like to do on the side bar")   


def scale_data(df, columns):
    scaler = StandardScaler()
    for column in columns:
        df[column] = scaler.fit_transform(df[[column]])
    time.sleep(1)
    st.write(df)
    st.success(f'✅ Successfully Scaled {columns}')
    st.session_state.to_clean = df  
    st.session_state.history_tracker.save_version(st.session_state.to_clean, action=f"Feature Scaling: {columns}")
    st.info("Clean data has been stored for other processes. Select what else you'll like to do on the side bar")


def one_hot_encoding(df,columns):  
    df_encoded = pd.get_dummies(df, columns = columns) 
    st.markdown('##### Instant View of One-Hot Encoded Data:') 
    st.write(df_encoded.head())
    df = df_encoded 
    st.session_state.to_clean = df  
    st.session_state.history_tracker.save_version(st.session_state.to_clean, action=f"BOne Hot Encoding: {columns}")
    st.info("Clean data has been stored for other processes. Select what else you'll like to do on the side bar")   


def label_encoding(df, columns):
    le = LabelEncoder()
    for column in columns:
        df[column] = le.fit_transform(df[column])
    st.markdown('##### Instant View of Label Encoded Data:') 
    st.write(df.head())
    st.session_state.to_clean = df  
    st.session_state.history_tracker.save_version(st.session_state.to_clean, action=f"Label Encoding: {column}")
    st.info("Clean data has been stored for other processes. Select what else you'll like to do on the side bar")

def add_columns(df):
    st.markdown('#### $Data$ $Preview$')
    st.dataframe(df.head())

    st.markdown('---')
    st.markdown('##### Write Code Logic Below to Add Columns')
    code = st_ace('Write New Column Logic Here (Remember to comment this line out):', language='python', theme= "monokai")

    if st.button('Preview Code Execution'):
        try:
            # Create a temporary copy of df for preview
            df_preview = df.copy()
            exec_globals = {"df": df_preview}
            exec(code, exec_globals)
            df_preview = exec_globals["df"]  # Updated preview

            st.success('✅ Code executed successfully! Preview below:')
            st.dataframe(df_preview)  # Show preview only

        except Exception as e:
            st.error(f"❌ Error in code execution: {e}")

    if st.button('Apply Changes'):
        try:
            exec_globals = {"df": df}
            exec(code, exec_globals)
            df = exec_globals["df"]  # Apply changes permanently

            st.success('✅ Changes applied to DataFrame!')
            st.session_state.to_clean = df
            st.session_state.history_tracker.save_version(st.session_state.to_clean, action=f"Added columns with code")
            st.dataframe(df.head())  # Show updated DataFrame

        except Exception as e:
            st.error(f"❌ Error applying changes: {e}")