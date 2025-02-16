import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px
import re
from transformations import *
from Main import overview_data
from streamlit_ace import st_ace

st.markdown('# Clean Up Your Data 🧹🚿')

if 'clean_page' not in st.session_state:
    st.session_state.clean_page = 'one'

def proceed_to_clean_up_data():
    st.session_state.clean_page = 'two'        


def back_to_preview():
    st.session_state.clean_page = 'one'   


def clean_data(datasets):
    if datasets is not None:
       st.markdown(f'##### The following tables have been stored for you for use:')
       for i,data in enumerate(datasets.keys()):
           st.markdown(f'###### {i+1}. {data}  \n')   
       selected = st.selectbox('Choose a table to clean:', options=list(datasets.keys()), index=None) 
       if selected:
           df = datasets[selected]
           st.session_state.to_clean = df
           st.markdown('### See Overview Before Cleaning:')
           overview_data(df)
           st.markdown('#### $More$ $Details$')
           col1, col2 = st.columns(2)
           with col1:
             st.markdown('#### Duplicated Row Count:')
             st.markdown(f'# ${df.duplicated().sum()}$')

           with col2:  
             st.markdown('#### Null Value Count:')
             st.markdown(f'# ${df.isnull().sum().sum()}$')

           categorical = []
           continuous = []
           date = []
           other = []
           unique_index = df.nunique().index
           unique_values = df.nunique().values
           for i, unique in enumerate(unique_index):
               if unique_values[i] <= 8:
                   categorical.append(unique)

               elif unique_values[i] > 8:
                   if df[unique].dtype == 'O' :
                        other.append(unique)

                   elif pd.api.types.is_datetime64_any_dtype(df[unique]):
                        date.append(unique)

                   else: continuous.append(unique)
           st.session_state.to_clean_cont = continuous
           st.session_state.to_clean_cat = categorical  
           st.session_state.to_clean_str = other  
           st.session_state.to_clean_date = date
           st.markdown('#### $Variable$ $Count$:')  
           col3, col4 = st.columns(2)
           with col3:
              st.markdown('#### Categorical:')
              st.markdown(f'# ${len(categorical)}$')

           with col4:
              st.markdown('#### Continuous:')
              st.markdown(f'# ${len(continuous)}$')  

           col5, col6 = st.columns(2)   
           with col5:
              st.markdown('#### Strings:')
              st.markdown(f'# ${len(other)}$')

           with col6:
                st.markdown('#### Dates:')
                st.markdown(f'# ${len(date)}$')

           
           st.markdown('### Columns with Null Values:')
           null_count = {}
           null_index = df.isnull().sum().index
           null_values = df.isnull().sum().values
           for j ,null in enumerate(null_index):
              null_count[null] = int(null_values[j])
           st.write(null_count)
           st.session_state.null_count = null_count 

           clean_btn = st.button('Proceed to Clean Up Data')
           if clean_btn:
            proceed_to_clean_up_data()

          
def clean_up_data():
        st.markdown('---')
        st.markdown('### Clean Up Your Data:')
        df = st.session_state.to_clean
        null_val = st.session_state.null_count
        continuous = st.session_state.to_clean_cont
        categorical = st.session_state.to_clean_cat
        date = st.session_state.to_clean_date
        other = st.session_state.to_clean_str
        
        method = st.sidebar.selectbox('What shall we do first:', options=['Drop Rows', 
                                                                      'Drop Columns', 
                                                                      'Handle Missing Values',  
                                                                      'Transformations', 
                                                                      'Drop Duplicated Rows', 
                                                                      'Rename Columns', 
                                                                      'Alter values within columns'], index=None)


        if method == 'Drop Rows':
           drop_rows(df)

        elif method == 'Drop Duplicated Rows':
            drop_duplicates(df) 

        elif method == 'Drop Columns':
            drop_columns(df)  

        elif method == 'Handle Missing Values' :
           handle_missing_values(df, null_val)

        elif method == 'Transformations':   
            transformation(df, categorical, continuous, other, date)

        elif method == 'Rename Columns':
            rename_columns(df)

        elif method == 'Alter values within columns':
            alter_values(df)
                 
        elif method == None:
            st.info('Choose what you want to do on the side bar')    


        st.markdown('---')
        back_btn = st.button('Go Back to Preview')    
        if back_btn:
            back_to_preview()


def drop_rows(df):
    row_count = st.number_input('Enter the number of rows to drop:', min_value=1, max_value=1000)
    if row_count > 0:
        st.markdown('#### Preview')
        st.write(df.head())
        st.write({f'Current number of rows: {len(df)}'})

        mode = st.selectbox('Choose how yo want to drop these rows:', options=[f'First {row_count}', f'Randomn {row_count}'], index=None)
        if mode is not None:
            if mode == f'First {row_count}':
              df = df.iloc[row_count:].reset_index(drop=True)

            elif mode == f'Randomn {row_count}':
              random_rows = df.sample(n=row_count, random_state=42)  
              df = df.drop(random_rows.index).reset_index(drop=True)

            time.sleep(2)  
            st.markdown('#### After Drop')
            st.dataframe(df.head())
            st.write({f'Current number of rows: {len(df)}'}) 
            st.success('✅ Successfully Dropped Rows')    
            st.session_state.to_clean = df
            st.info("Clean data has been stored for other processes. Select what else you'll like to do on the side bar")       
    
def drop_duplicates(df):
    st.markdown('#### $Duplicated$ $Rows$')
    duplicate = df[df.duplicated()]

    if len(duplicate) > 0:
      st.dataframe(duplicate)
      st.write({f"Duplicated Rows Count: {len(duplicate)}"})
      drop_duplicates_btn = st.button('Drop Duplicates')
      if drop_duplicates_btn:
          df = df.drop_duplicates().reset_index(drop=True)
          st.success('✅ Successfully Dropped Duplicates')
          st.session_state.to_clean = df
          st.info("Clean data has been stored for other processes. Select what else you'll like to do on the side bar")
    
    else: st.info('No duplicated rows found')

def drop_columns(df):
    st.markdown('#### $Data$ $Preview$')
    st.dataframe(df.head())
    columns = st.multiselect('Choose columns to drop:', options=df.columns, default=None)

    if columns:     
        confirm = st.radio(f'Are you sure you want to drop {columns}', options=['Yes', 'No'], index = None)
        if confirm == 'Yes':
          time.sleep(2)
          df = df.drop(columns, axis=1)
          st.success('✅ Successfully Dropped Columns')
          st.session_state.to_clean = df
          st.markdown('#### $After$ $Drop$')
          st.dataframe(df.head())
          st.write(f'Current Columns: {[df.columns]}')
          st.info("Clean data has been stored for other processes. Select what else you'll like to do on the side bar") 

        elif confirm == 'No' : 
            st.info('Select another set of columns to drop')

def handle_missing_values(df, null_val):
    st.markdown('#### Preview')
    st.write(df.head())
    st.markdown('#### Percentage of Null Values')
    null_percentage = round(df.isnull().sum().sum() / len(df) * 100, 2)
    st.markdown(f'# {null_percentage}%')
    st.markdown('#### Null values per column: ')
    st.write(null_val)
    null_list = []
    for col, null in null_val.items():
        if null > 0:
            null_list.append(col)
    missing = st.selectbox('Choose columns to handle missing values:', options=null_list, index=None)
    if missing:
        st.markdown(f"#### Rows Where  ${missing}$  is Missing")
        st.dataframe(df[df[missing].isnull()].head())
        method = st.selectbox('Choose how you want to handle missing values:', options=['Drop Rows', 'Drop Columns', 'Impute Values'], index=None)

        if method == 'Drop Rows':
             confirm = st.radio(f'Are you sure yo want to drop all rows where {missing} is missing', options=['Yes', 'No'], index = None)
             if confirm == 'Yes':
               time.sleep(2)
               df.dropna(subset=missing, inplace=True)
               st.success(f'✅ Successfully Dropped Rows with missing {missing} values')
               st.session_state.to_clean = df
               st.info("Clean data has been stored for other processes. Select what else you'll like to do on the side bar")

        if method == 'Drop Columns':
             confirm = st.radio(f'Are you sure yo want to drop the {missing} column', options=['Yes', 'No'], index = None)
             if confirm == 'Yes':
                 time.sleep(1)
                 df.drop(missing, axis=1, inplace = True)
                 st.success(f'✅ Successfully Dropped Feature {missing}')
                 st.session_state.to_clean = df
                 st.info("Clean data has been stored for other processes. Select what else you'll like to do on the side bar")

        if method == 'Impute Values':
            method = st.selectbox('Choose an aggregate to impute missing values with:', options=['Mean', 'Median', 'Mode', 'Constant Value'], index=None)
            if method is not None:
                if method == 'Mean':
                    df[missing] = df[missing].fillna(df[missing].mean())
                    st.success(f'✅ Successfully Imputed Missing Values with {method}')
                    st.session_state.to_clean = df
                    st.info("Clean data has been stored for other processes. Select what else you'll like to do on the side bar")
                
                elif method == 'Median':
                    df[missing] = df[missing].fillna(df[missing].median())
                    st.success(f'✅ Successfully Imputed Missing Values with {method}')
                    st.session_state.to_clean = df
                    st.info("Clean data has been stored for other processes. Select what else you'll like to do on the side bar")
                
                elif method == 'Mode':
                    df[missing] = df[missing].fillna(df[missing].mode()[0])
                    st.success(f'✅ Successfully Imputed Missing Values with {method}')
                    st.session_state.to_clean = df
                    st.info("Clean data has been stored for other processes. Select what else you'll like to do on the side bar")
                
                elif method == 'Constant Value':
                    constant = st.text_input('Enter the constant value:', value='')
                    if len(constant) > 0:  # Only fill if the user has entered a value.
                        btn = st.button('Apply Constant Value')
                        if btn:
                           df[missing] = df[missing].fillna(constant)
                           st.success(f'✅ Successfully Imputed Missing Values with {method}')
                           st.session_state.to_clean = df
                           st.info("Clean data has been stored for other processes. Select what else you'll like to do on the side bar")

def transformation(df,categorical, continuous, other, dates):
    st.warning('⚠️ This section is for data transformation. Please note that preferably, all other data wrangling techniques should be carried out to satisfaction before transformation.')
    st.markdown('### Data Preview')
    st.write(df.head())
    st.write({f'Current number of rows: {len(df)}'})
    st.subheader('$Categorical$ $Variables$')
    st.write(categorical)
    st.subheader('$Continuous$ $Variables$')
    st.write(continuous)
    st.subheader('$Other$ $Variables$')
    st.write(other)
    st.subheader('$Datetime$ $Variables$')
    st.write(dates)
    st.markdown('---')

    transformation_type = st.sidebar.selectbox('Choose a transformation type:', options=['Binning', 'Change Data Type', 'Feature Scaling', 'Label Encoding', 'One-Hot Encoding'], index=None)
    if transformation_type is not None:
        if transformation_type == 'Change Data Type':
            column = st.selectbox('Choose the column to change data type:', options=df.columns, index=None)

            if column is not None:
                st. write({f'Current Datatype: {df[column].dtype}'})
                convert_to = st.selectbox('Choose the data type to convert to:', options=['int', 'float', 'object', 'datetime'], index=None)
                change_data_types(df, column, convert_to)
             
        
        elif transformation_type == 'Binning':
            column = st.selectbox('Choose the column to bin:', options=continuous, index=None)
            if column is not None:
                binning(df, column)

    
        elif transformation_type == 'Feature Scaling':
            column = st.selectbox('Choose the column to scale:', options=continuous, index=None)
            if column is not None:
                scale_data(df, column)

        elif transformation_type == 'One-Hot Encoding':
            column = st.multiselect('Choose the columns to encode:', options=categorical, default = None)
            one_hot_encoding(df, column)

        elif transformation_type == 'Label Encoding':
            column = st.multiselect('Choose the columns to encode:', options=categorical, default = None)
            label_encoding(df, column)    

    else : st.info('Go on to select a transformation type from the sidebar!')

def rename_columns(df):
    st.markdown('#### $Data$ $Preview$')
    st.dataframe(df.head())
    column = st.selectbox('Choose column to rename:', options=df.columns, index=None)
    

    if column: 
        new_name = st.text_input('Enter new column name:', value='')
        if new_name:    
           confirm = st.radio(f'Are you sure you want to rename {column} to {new_name}', options=['Yes', 'No'], index = None)
           if confirm == 'Yes':
              df.rename(columns={column: new_name}, inplace=True)
              st.success(f'✅ Successfully Renamed Column {column} to {new_name}')
              st.session_state.to_clean = df
              st.info("Clean data has been stored for other processes. Select what else you'll like to do on the side bar")

def alter_values(df):
    st.markdown('#### $Data$ $Preview$')
    st.dataframe(df.head())
    column = st.selectbox('Choose column to alter:', options=df.columns, index=None)
    
    if column:
        st.markdown('#### $All$ $Column$ $Values$')
        st.write(df[column].value_counts().to_dict())
        format = st.selectbox('How will you like to format your data?', ['Code', 'Text' ], index=None)
        if format == 'Text':
          value_to_alter = st.text_input(f'Enter the value within {column} to alter:', value='').strip()
         # Ensure correct data type before using it
          if value_to_alter:
           column_dtype = df[column].dtype

           try:
              if column_dtype == 'int64':
                value_to_alter = int(value_to_alter)

              elif column_dtype == 'float64':
                value_to_alter = float(value_to_alter)

              elif column_dtype == 'datetime64[ns]':
                  value_to_alter = pd.to_datetime(value_to_alter)

                  if pd.isna(value_to_alter):
                    raise ValueError("Invalid date format. Use YYYY-MM-DD or similar.")
                    # No conversion needed for categorical or object (string) types
            
              # Check if the value exists in the column
              if value_to_alter in df[column].values:
                st.success(f"✅ Valid value: {value_to_alter} in {column} (Type: {column_dtype})")
                new_value = st.text_input(f'Prompt value to replace {value_to_alter} in {column}', value='').strip()
                if new_value is not None:
                        confirm = st.radio(f'Are you sure you want to replace {value_to_alter} with {new_value} in {column}', options=['Yes', 'No'], index = None)
                        if confirm == 'Yes':
                           df[column] = df[column].replace(value_to_alter, new_value)
                           st.success(f'✅ Successfully Altered Value {value_to_alter} with {new_value} in {column}')
                           st.session_state.to_clean = df
                           st.info("Clean data has been stored for other processes. Select what else you'll like to do on the side bar")

              else:
                st.warning(f"⚠️ Value '{value_to_alter}' not found in column '{column}'")

           except ValueError:
              st.error(f"❌ Invalid data type! Expected {column_dtype}, but got '{value_to_alter}'")    
        
        elif format == 'Code':
          code = st_ace('Write Transformative Code Here (Remember to comment this line out):', language='python', theme= "monokai")

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
                    st.dataframe(df.head())  # Show updated DataFrame

                except Exception as e:
                    st.error(f"❌ Error applying changes: {e}")

    
if "data" not in st.session_state:
    st.info('Upload data on the main page.') 

else:    
    if st.session_state.clean_page == 'one':
      clean_data(st.session_state.data) 

    elif st.session_state.clean_page == 'two':
       st.session_state.data['Cleaned data'] = st.session_state.to_clean
       clean_up_data()
       