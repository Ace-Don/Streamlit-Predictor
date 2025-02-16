import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from models import *

if 'model_page' not in st.session_state:
    st.session_state.model_page = 'one'

def proceed_to_model_data():
    st.session_state.model_page = 'two'        


def back_to_model_pre():
    st.session_state.model_page = 'one'  

st.title('Make Instant Predictions Based on Your Data🔮')

def modelling(datasets):
    # st.warning('This is a prototype version and does not include advanced machine learning models or real-time data updates.')
    st.warning('⚠️ For the most optimal performance and results, ensure your data is clean and organized before fitting it to any model.')
    if datasets is not None:
       st.markdown(f'##### The following tables have been stored for you for use:')
       for i,data in enumerate(datasets.keys()):
           st.markdown(f'###### {i+1}. {data}  \n')   
       selected = st.selectbox('Choose a table to train and test your model with:', options=list(datasets.keys()), index=None) 
       if selected:
           df = datasets[selected]
           st.markdown('### $Data$ $Preview:$')
           st.dataframe(df.head())
           target = st.selectbox('Choose your target column:', options=df.columns, index = None)
           if target is not None:
                X = df.drop(target, axis=1)
                Y = df[target]
                st.write('#### Predictor Sample:')
                st.dataframe(X.sample(5, random_state=42))
                st.write('#### Corresponding Target Sample:')
                st.dataframe(Y.sample(5, random_state=42))

                st.markdown('---')
                st.markdown('### $Splitting$ $Data$ : $Phase$ $1$')
                st.info('Let\'s split your data into training and testing sets. Note that the training set will be futher split into training and validation sets.')
                st.write({f'Dataset Size: {len(df)}'})
                train_sample_number = st.select_slider('Guage the number of samples for training set:',  options=list(range(1, len(df) + 1)), value= int(len(df)*0.75))
                if train_sample_number > 0:
                    split_confirm = st.radio(f'Split {train_sample_number} Rows of Data', ['Yes', 'No'], index=None)

                    if split_confirm == 'Yes':
                       train = df.sample(train_sample_number, random_state=42).reset_index(drop=True)
                       test = df.drop(train.index).reset_index(drop=True)

                       ############################################################
                       st.session_state.test_data = {'X' : test.drop(target, axis=1), 
                                                     'Y' : test[target]
                                                    }
                       st.success(f'✅ Split Successfull. Training set size: {len(train)}. Test set size: {len(test)}')

                       st.markdown('---')
                       st.markdown('### $Splitting$ $Data$ : $Phase$ $2$')
                       st.info('Now let\'s split your training set further into training and validation sets.')
                       x = train.drop(target, axis=1)
                       y = train[target]
                       val_ratio = st.select_slider('Enter ratio for validation set:', options = np.round(np.linspace(0.1, 0.9, 9),2), value=0.2)
                       if val_ratio > 0:
                            val_confirm = st.radio(f'Split {val_ratio*100} % of Training set data', ['Yes', 'No'], index=None)
                            if val_confirm == 'Yes':
                               x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

                               #################################################################
                               st.session_state.train_data = {'X_train' : x_train, 
                                                              'Y_train' : y_train, 
                                                              'X_val' : x_val, 
                                                              'Y_val' : y_val}
                               st.success(f'✅ Split Successfull. Train set size: {len(x_train)}. Validation set size: {len(x_val)}.')
                               
                               st.markdown('---')
                               proceed_button = st.button('Proceed to Model Training')
                               if proceed_button:
                                   proceed_to_model_data()
                                   

                               # TODO: Implement machine learning models here.

                            else: st.info('Confirm ratio for validation set')   
                    
                    else : st.info('Confirm number of rows to split data')

                # train_set = 

def choose_model():
    st.markdown('---')
    st.subheader('Choose a machine learning model to train your data:')
    st.info('This section is for selecting and training machine learning models. Please note that advanced machine learning models will be implemented in the next phase.')
    task = st.selectbox('What kind of task do you want to perform?', ['Classification', 'Regression'], index = None)
    if task is not None:
        if task == 'Classification':
            st.sidebar.subheader('Choose a classification model:')
            st.markdown('#### Quick View of Training Data:')

            #####################################################################
            st.markdown('###### Predictors')
            x_tr =st.session_state.train_data['X_train']
            st.write(x_tr.head())
            st.markdown('###### Target')
            y_tr =st.session_state.train_data['Y_train']
            st.write(y_tr.head())
            x_ts = st.session_state.train_data['X_val']
            y_ts = st.session_state.train_data['Y_val']
            #####################################################################

            model = st.sidebar.selectbox('', ['Logistic Regression', 'Support Vector Machines', 'Random Forest', 'XGBoost', 'CatBoost', 'K-Neighbors', 'Decision Trees'], index = None)
            if model is not None:
                st.markdown('---')
                st.markdown(f'#### {model}')
                classification(model, x_tr, y_tr, x_ts, y_ts)

        elif task == 'Regression':
            st.sidebar.subheader('Choose a regression model:')
            st.markdown('#### Quick View of Training Data:')

            #####################################################################
            st.markdown('###### Predictors')
            x_tr =st.session_state.train_data['X_train']
            st.write(x_tr.head())
            st.markdown('###### Target')
            y_tr =st.session_state.train_data['Y_train']
            st.write(y_tr.head())
            x_ts = st.session_state.train_data['X_val']
            y_ts = st.session_state.train_data['Y_val']
            #####################################################################

            model = st.sidebar.selectbox('', ['Linear Regression', 'Decision Trees'], index = None)
            if model is not None:
                st.markdown('---')
                st.markdown(f'#### {model}')
                regression(model, x_tr, y_tr, x_ts, y_ts)


    else: st.info('Select a modelling task')


    st.markdown('---')
    back_btn = st.button('Back to Model Preparation')    
    if back_btn:
            back_to_model_pre()


def  classification(model, x_tr, y_tr, x_ts, y_ts):
        if model == 'Logistic Regression': log_reg_model(x_tr, y_tr, x_ts, y_ts)
                        
        elif model =='Decision Trees': decision_tree_model(x_tr, y_tr, x_ts, y_ts)

        elif model == 'Random Forest': random_forest_model(x_tr, y_tr, x_ts, y_ts)

        elif model == 'XGBoost': xgboost_model(x_tr, y_tr, x_ts, y_ts)

        elif model == 'CatBoost': catboost_model(x_tr, y_tr, x_ts, y_ts)
        
        elif model == 'K-Neighbors': knn_model(x_tr, y_tr, x_ts, y_ts)

        elif model == 'Support Vector Machines': svm_model(x_tr, y_tr, x_ts, y_ts)

def regression(model, x_tr, y_tr, x_ts, y_ts):
    if model == 'Linear Regression': linear_regression(x_tr, y_tr, x_ts, y_ts)
                        
    elif model == 'Decision Trees': decision_tree_model_reg(x_tr, y_tr, x_ts, y_ts)


if "data" not in st.session_state:
    st.info('Upload data on the main page.') 

else:    
    if st.session_state.model_page == 'one':
      modelling(st.session_state.data)

    elif st.session_state.model_page == 'two':
       choose_model()










