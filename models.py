import streamlit as st
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder, LabelEncoder
from sklearn.metrics import jaccard_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import time
import pandas as pd


def oversampler_func(y, input):
       # account for imbalance in the dataset
    value_count = y.value_counts()
    if len(value_count) == 2:
        for index in value_count.index:
            if (value_count[index]*3) < value_count.max():
                st.info(f'The target column is imbalanced and class {index} is of the minority class')
                confirm = st.radio('Will you like incorporate oversampling for the minority class in the pipeline?', ['Yes', 'No'], index=None)
                if confirm == 'Yes':
                    samp_type = st.selectbox('Choose oversampling method:', ['ADASYN', 'SMOTE'], index=None)
                    if samp_type == 'ADASYN':
                        st.sidebar.markdown('---')
                        st.info('Choose ADASYN parameters on the sidebar')
                        st.sidebar.subheader('Choose ADASYN parameters:')
                        neighbors = st.sidebar.number_input('Enter the number of neighbors for ADASYN:', min_value=1, max_value=10, value = 5)
                        samp_ratio = st.sidebar.number_input('Enter the sampling ratio (i.e the ratio ypu want the minority class compared to the majority to be):', min_value=0.1, max_value=1.0, value=0.8)
                        if neighbors and samp_ratio:
                           oversampler = ADASYN(random_state=42, n_neighbors = neighbors, sampling_strategy=samp_ratio)
                           input.append(('oversampler', oversampler))

                    elif samp_type == 'SMOTE':
                        st.sidebar.markdown('---')
                        st.info('Choose SMOTE parameters on the sidebar')
                        st.sidebar.subheader('Choose SMOTE parameters:')
                        neighbors = st.sidebar.number_input('Enter the number of neighbors for SMOTE:', min_value=1, max_value=100)
                        samp_ratio = st.sidebar.number_input('Enter the sampling ratio (i.e the ratio ypu want the minority class compared to the majority to be):', min_value=0.1, max_value=1.0, value=1.0)
                        if neighbors and samp_ratio:
                           oversampler = SMOTE(random_state=42, k_neighbors = neighbors, sampling_strategy=samp_ratio)
                           input.append(('oversampler', oversampler))
                
                return confirm


def model_train_classification(input, params, X_train, y_train, X_test, y_test):
    time.sleep(0.5)
    pipeline = Pipeline(input)
    st.success('✅ Sucessfully built pipeline!')
    grid_search = GridSearchCV(pipeline, params, cv=5, scoring='roc_auc', n_jobs=1, error_score= 'raise')
    check = st.checkbox('Check me to commence training your model.')
    if check:
        time.sleep(1)
        st.warning('⚠️ Training in progress... Do not interrupt the process.')
        grid_search.fit(X_train, y_train)
        st.markdown('---')

        ########################################################################
        st.balloons()
        st.success('Successfully trained model')
        st.markdown('##### In-Sample Evaulation Score')
        st.write(grid_search.best_score_)
        st.markdown('Most Optimal Parameter Combinations')
        st.write(grid_search.best_params_) 
        #########################################################################

        st.info('Remember, we made a validation set! You can see how your model performs with it.')
        test_val = st.radio('Do you want to see how your model performs with your validation set?', ['Yes', 'No'], index = None)
        if test_val == 'Yes':
            st.markdown('##### Out-of-Sample Evaulation Score')
            y_pred = grid_search.predict(X_test)
            st.markdown(f'Accuracy Score: {round(accuracy_score(y_test, y_pred), 2)}')
            st.markdown(f"F1 Score: {round(f1_score(y_test, y_pred, average='macro'), 2)}")
            st.markdown(f"Jaccard_score: {round(jaccard_score(y_test, y_pred, average='macro'), 2)}")

    else:
        st.stop()


def model_train_regression(input, params, X_train, y_train, X_test, y_test):
    time.sleep(0.5)
    pipeline = Pipeline(input)
    st.success('✅ Sucessfully built pipeline!')
    grid_search = GridSearchCV(pipeline, params, cv=5, scoring='r2', n_jobs=1, error_score= 'raise')
    check = st.checkbox('Check me to commence training your model.')
    if check:
        time.sleep(1)
        st.warning('⚠️ Training in progress... Do not interrupt the process.')
        grid_search.fit(X_train, y_train)
        st.markdown('---')

        ########################################################################
        st.balloons()
        st.success('Successfully trained model')
        st.markdown('##### In-Sample Evaulation Score')
        st.write(grid_search.best_score_)
        st.markdown('Most Optimal Parameter Combinations')
        st.write(grid_search.best_params_) 
        #########################################################################

        st.info('Remember, we made a validation set! You can see how your model performs with it.')
        test_val = st.radio('Do you want to see how your model performs with your validation set?', ['Yes', 'No'], index = None)
        if test_val == 'Yes':
            st.markdown('##### Out-of-Sample Evaulation Score')
            y_pred = grid_search.predict(X_test)
            st.markdown(f'R2 Score: {round(r2_score(y_test, y_pred), 2)}')
            st.markdown(f"Mean Squared Error: {round(mean_squared_error(y_test, y_pred), 2)}")

    else:
        st.stop()
    

def log_reg_model(X_train, y_train, X_test, y_test):
  st.info('Know that a Logistic Regression model only works with numerical data, soo make sure your transformations are performed as due.')
  scale_list, one_hot_list, label_list = transform_column_choice(X_train)
  transform_check = st.checkbox('Check if you have selected what columns to apply your desired transformation steps to.')
 


  if transform_check:
    preprocessor = ColumnTransformer(
    transformers=[('num', StandardScaler(), scale_list), 
        ('cat', OneHotEncoder(), one_hot_list),
        ('label', LabelEncoder(), label_list)
        ])

    input = [('preprocessor', preprocessor)]
    # hyperparameter grid
    param_grid = {'model__C': [0.01, 0.1, 1, 10, 100],
                  'model__penalty': ['l1', 'l2'],
                  'model__solver': ['liblinear'],
                  'model__class_weight': [None , 'balanced']
                  }
    
    sample = oversampler_func(y_train, input)

    if sample == 'Yes' and input[-1][0] == 'oversampler': 
        ok = st.radio('Are you satisfied with your oversampling settings?', options=['Yes', 'No'], index = None )
        if ok == 'Yes':
            input.append(('model', LogisticRegression()))
            model_train_classification(input,param_grid,X_train, y_train, X_test, y_test)
            
                
        elif ok =='No' : 
            st.info('Adjust settings for oversampler on the sidebar')  
    
    elif sample == 'No': 
        input.append(('model', LogisticRegression()))
        model_train_classification(input, param_grid,X_train, y_train, X_test, y_test)

  else : st.info('If you do not desire to add transformation steps to the pipeline, you should still check the box to proceed with training')


def decision_tree_model(X_train, y_train, X_test, y_test):
  scale_list, one_hot_list, label_list = transform_column_choice(X_train)
  transform_check = st.checkbox('Check if you have selected what columns to apply your desired transformation steps to.')


  if transform_check:
    preprocessor = ColumnTransformer(
    transformers=[('num', StandardScaler(), scale_list), 
        ('cat', OneHotEncoder(), one_hot_list),
        ('label', LabelEncoder(), label_list)
        ])

    input = [('preprocessor', preprocessor)]

    # hyperparameter grid
    param_grid = {'model__criterion': ['gini', 'entropy'],
                  'model__max_depth': [None, 5, 10, 15, 20],
                  'model__min_samples_split': [2, 5, 10, 15, 20],
                  'model__min_samples_leaf': [1, 2, 5, 10, 15]
                  }

    sample = oversampler_func(y_train, input)

    if sample == 'Yes' and input[-1][0] == 'oversampler': 
        ok = st.radio('Are you satisfied with your oversampling settings?', options=['Yes', 'No'], index = None )
        if ok == 'Yes':
            input.append(('model', DTC()))
            model_train_classification(input,param_grid,X_train, y_train, X_test, y_test)
            
                
        elif ok =='No' : 
            st.info('Adjust settings for oversampler on the sidebar')  
    
    elif sample == 'No': 
        input.append(('model', DTC()))
        model_train_classification(input, param_grid,X_train, y_train, X_test, y_test)
    
  else : st.info('If you do not desire to add transformation steps to the pipeline, you should still check the box to proceed with training')

def random_forest_model(X_train, y_train, X_test, y_test):
  scale_list, one_hot_list, label_list = transform_column_choice(X_train)
  transform_check = st.checkbox('Check if you have selected what columns to apply your desired transformation steps to.')


  if transform_check:
    preprocessor = ColumnTransformer(
    transformers=[('num', StandardScaler(), scale_list), 
        ('cat', OneHotEncoder(), one_hot_list),
        ('label', LabelEncoder(), label_list)
        ])

    input = [('preprocessor', preprocessor)]

    # hyperparameter grid
    param_grid = {  'model__n_estimators':[50,100,200],
                    'model__max_depth':[None,5,10,20,30],
                    'model__min_samples_split':[2,5,10],
                    'model__min_samples_leaf':[1,2,4],
                    'model__bootstrap' : [True, False],
                    'model__class_weight': [None , 'balanced']
    }

    sample = oversampler_func(y_train, input)

    if sample == 'Yes' and input[-1][0] == 'oversampler': 
        ok = st.radio('Are you satisfied with your oversampling settings?', options=['Yes', 'No'], index = None )
        if ok == 'Yes':
            input.append(('model', RandomForestClassifier()))
            model_train_classification(input,param_grid,X_train, y_train, X_test, y_test)
            
                
        elif ok =='No' : 
            st.info('Adjust settings for oversampler on the sidebar')  
    
    elif sample == 'No': 
        input.append(('model', RandomForestClassifier()))
        model_train_classification(input, param_grid,X_train, y_train, X_test, y_test)

  else : st.info('If you do not desire to add transformation steps to the pipeline, you should still check the box to proceed with training')


def xgboost_model(X_train, y_train, X_test, y_test):
  scale_list, one_hot_list, label_list = transform_column_choice(X_train)
  transform_check = st.checkbox('Check if you have selected what columns to apply your desired transformation steps to.')


  if transform_check:
    preprocessor = ColumnTransformer(
    transformers=[('num', StandardScaler(), scale_list), 
        ('cat', OneHotEncoder(), one_hot_list),
        ('label', LabelEncoder(), label_list)
        ])

    input = [('preprocessor', preprocessor)]

    # hyperparameter grid
    param_grid = {  'model__n_estimators':[50,100,200],
                    'model__max_depth':[None,5,10,20,30],
                    'model__learning_rate':[0.01,0.1,0.5],
                    'model__gamma':[0,0.1,1],
                   'model__subsample':[0.5,1],
                   'model__colsample_bytree':[0.5,1],
                   'model__min_child_weight':[1,5,10],
                   'model__class_weight': [None , 'balanced']
    }

    sample = oversampler_func(y_train, input)

    if sample == 'Yes' and input[-1][0] == 'oversampler': 
        ok = st.radio('Are you satisfied with your oversampling settings?', options=['Yes', 'No'], index = None )
        if ok == 'Yes':
            input.append(('model', XGBClassifier(eval_metric='logloss',  tree_method='hist', random_state=42)))
            model_train_classification(input,param_grid,X_train, y_train, X_test, y_test)
            
                
        elif ok =='No' : 
            st.info('Adjust settings for oversampler on the sidebar')  
    
    elif sample == 'No': 
        input.append(('model', XGBClassifier()))
        model_train_classification(input, param_grid,X_train, y_train, X_test, y_test)

  else : st.info('If you do not desire to add transformation steps to the pipeline, you should still check the box to proceed with training')


def svm_model(X_train, y_train, X_test, y_test):
  scale_list, one_hot_list, label_list = transform_column_choice(X_train)
  transform_check = st.checkbox('Check if you have selected what columns to apply your desired transformation steps to.')


  if transform_check:
    preprocessor = ColumnTransformer(
    transformers=[('num', StandardScaler(), scale_list), 
        ('cat', OneHotEncoder(), one_hot_list),
        ('label', LabelEncoder(), label_list)
        ])

    input = [('preprocessor', preprocessor)]
    # hyperparameter grid
    param_grid = { 'model__C': [0.1, 1, 10, 100, 1000],
                  'model__gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'model__kernel': ['linear', 'rbf']}
    
    sample = oversampler_func(y_train, input)
    
    if sample == 'Yes' and input[-1][0] == 'oversampler':
        ok = st.radio('Are you satisfied with your oversampling settings?', options=['Yes', 'No'], index = None )
        if ok == 'Yes':
            input.append(('model', SVC()))
            model_train_classification(input,param_grid,X_train, y_train, X_test, y_test)
            
                
        elif ok =='No' : 
            st.info('Adjust settings for oversampler on the sidebar')

    elif sample == 'No': 
        input.append(('model', SVC()))
        model_train_classification(input, param_grid,X_train, y_train, X_test, y_test)

  else : st.info('If you do not desire to add transformation steps to the pipeline, you should still check the box to proceed with training')


def knn_model(X_train, y_train, X_test, y_test):
  scale_list, one_hot_list, label_list = transform_column_choice(X_train)
  transform_check = st.checkbox('Check if you have selected what columns to apply your desired transformation steps to.')

  if transform_check:
    preprocessor = ColumnTransformer(
    transformers=[('num', StandardScaler(), scale_list), 
        ('cat', OneHotEncoder(), one_hot_list),
        ('label', LabelEncoder(), label_list)
        ])

    input = [('preprocessor', preprocessor)]
    # hyperparameter grid
    param_grid = { 'model__n_neighbors': [3, 5, 7, 9, 11],
                  'model__weights': ['uniform', 'distance'],
                  'model__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                  'model__metric': ['euclidean', 'manhattan', 'minkowski']
                  }
    
    sample = oversampler_func(y_train, input)
    
    if sample == 'Yes' and input[-1][0] == 'oversampler':
        ok = st.radio('Are you satisfied with your oversampling settings?', options=['Yes', 'No'], index = None )
        if ok == 'Yes':
            input.append(('model', KNeighborsClassifier()))
            model_train_classification(input,param_grid,X_train, y_train, X_test, y_test)
            
                
        elif ok =='No' :
            st.info('Adjust settings for oversampler on the sidebar')

    elif sample == 'No':
        input.append(('model', KNeighborsClassifier()))
        model_train_classification(input, param_grid,X_train, y_train, X_test, y_test)

  else : st.info('If you do not desire to add transformation steps to the pipeline, you should still check the box to proceed with training')
        

def catboost_model(X_train, ytrain, X_test, y_test):
  scale_list, one_hot_list, label_list = transform_column_choice(X_train)
  transform_check = st.checkbox('Check if you have selected what columns to apply your desired transformation steps to.')


  if transform_check:
    preprocessor = ColumnTransformer(
    transformers=[('num', StandardScaler(), scale_list), 
        ('cat', OneHotEncoder(), one_hot_list),
        ('label', LabelEncoder(), label_list)
        ])

    input = [('preprocessor', preprocessor)]
    # hyperparameter grid
    param_grid = {  'model__iterations': [100, 200, 300],
                    'model__learning_rate': [0.01, 0.1, 0.2],
                    'model__depth': [4, 6, 8],
                    'model__l2_leaf_reg': [1, 3, 5],
                    'model__subsample': [0.6, 0.8, 1.0],
                    'model__colsample_bylevel': [0.5, 0.7, 1.0]
    }
    
    sample = oversampler_func(ytrain, input)
    
    if sample == 'Yes' and input[-1][0] == 'oversampler':
        ok = st.radio('Are you satisfied with your oversampling settings?', options=['Yes', 'No'], index = None )
        if ok == 'Yes':
            input.append(('model', CatBoostClassifier(auto_class_weights = 'Balanced', loss_function='Logloss', verbose = 0)))
            model_train_classification(input,param_grid,X_train, ytrain, X_test, y_test)
            
                
        elif ok =='No' :
            st.info('Adjust settings for oversampler on the sidebar')

    elif sample == 'No':
        input.append(('model', CatBoostClassifier(auto_class_weights = 'Balanced', loss_function='Logloss', verbose = 0)))
        model_train_classification(input,param_grid,X_train, ytrain, X_test, y_test)    

  else : st.info('If you do not desire to add transformation steps to the pipeline, you should still check the box to proceed with training')
   

def decision_tree_model_reg(X_train, y_train, X_test, y_test):
  scale_list, one_hot_list, label_list = transform_column_choice(X_train)
  transform_check = st.checkbox('Check if you have selected what columns to apply your desired transformation steps to.')


  if transform_check:
    preprocessor = ColumnTransformer(
    transformers=[('num', StandardScaler(), scale_list), 
        ('cat', OneHotEncoder(), one_hot_list),
        ('label', LabelEncoder(), label_list)
        ])

    input = [('preprocessor', preprocessor)]

    # hyperparameter grid
    param_grid = {'model__criterion': ['absolute_error', 'friedman_mse', 'squared_error'],
                  'model__max_depth': [None, 5, 10, 15, 20],
                  'model__min_samples_split': [2, 5, 10, 15, 20],
                  'model__min_samples_leaf': [1, 2, 5, 10, 15]
                  }

    input.append(('model', DTR()))
    model_train_regression(input,param_grid,X_train, y_train, X_test, y_test)

  else : st.info('If you do not desire to add transformation steps to the pipeline, you should still check the box to proceed with training')
       


def linear_regression(X_train, y_train, X_test, y_test):
    scale_list, one_hot_list, label_list = transform_column_choice(X_train)
    transform_check = st.checkbox('Check if you have selected what columns to apply your desired transformation steps to.')


    if transform_check:
        preprocessor = ColumnTransformer(
        transformers=[('num', StandardScaler(), scale_list), 
        ('cat', OneHotEncoder(), one_hot_list),
        ('label', LabelEncoder(), label_list)
        ])

        input = [('preprocessor', preprocessor), ('poly', PolynomialFeatures())]
        param_grid = {'poly__degree': [i for i in range(1,10)], 
                  'poly__include_bias': [False, True]}
    

        reg = st.selectbox('Select what regularization modification you will like to incorporate', options=['Ridge', 'Lasso', 'None',], index=None)
        if reg == 'None':
           input.append(('model', LinearRegression()))
           param_grid.update({'model__fit_intercept': [True, False]})
           model_train_regression(input, param_grid, X_train, y_train, X_test, y_test)

        elif reg == 'Ridge':
           input.append(('model', Ridge()))
           param_grid.update({'model__fit_intercept': [True, False],
                           'model__alpha': [0.1, 1.0, 10.0, 100.0]})
           model_train_regression(input, param_grid, X_train, y_train, X_test, y_test)
        
        elif reg == 'Lasso':
           input.append(('model', Lasso()))
           param_grid.update({'model__fit_intercept': [True, False],
                           'model__alpha': [0.1, 1.0, 10.0, 100.0]})
           model_train_regression(input, param_grid, X_train, y_train, X_test, y_test)

    else : st.info('If you do not desire to add transformation steps to the pipeline, you should still check the box to proceed with training')

  

def transform_column_choice(data):
    cat, cont, date, other = data_type(data)
    continuous = [col for col in data.columns if col in cont]
    categorical = [col for col in data.columns if col in cat]
    st.sidebar.markdown('---')
    scaling = st.sidebar.multiselect('Select what colums you will want to scale in the model', options=continuous, default = None)
    one_hot = st.sidebar.multiselect('Select what categorical columns you will want to one-hot encode in the model', options=categorical, default = None)
    remaining_categorical = [col for col in categorical if col not in one_hot]
    label = st.sidebar.multiselect('Select what columns you will like to label encode in the model', options= remaining_categorical, default= None)
    return scaling, one_hot, label


def data_type(df):
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

    return categorical, continuous, date, other    