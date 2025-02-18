import streamlit as st
import pandas as pd
import numpy as np
import time

def merge_func():
  # Hide the upload button
  st.markdown("""
            <style>
              div[data-testid = "stFileUploader"]{display: none;}
            </style>
  """, unsafe_allow_html= True)


  sheets =  st.session_state.selected_sheets
  common_columns = st.session_state.common_columns
  file = st.session_state.excel_file
  st.markdown("### Merge your sheets to create a new one!")
  st.markdown(f'##### You selected the following sheets: *{sheets}*')
  max_select = 2
  selected_to_merge = st.multiselect('Which two sheets will you like to merge?', options=sheets)

  # Select at only two sheets to merge at a time
  if len(selected_to_merge) == 0:
    st.info('Please select two sheets to merge')

  elif len(selected_to_merge) == 1:
    st.warning('⚠️ Please select at least two sheets to merge')

  elif len(selected_to_merge) > max_select:
    st.warning(f'⚠️ You can only select {max_select} sheets to merge at a time')

  else: 
    selc_list = st.session_state.selected_to_merge = selected_to_merge  
    time.sleep(1)

    # Dynamically getting the common values between the selected sheets regardless of index order
    jup = set(common_columns.get(f'{selc_list[0]} & {selc_list[1]}', [])) | \
          set(common_columns.get(f'{selc_list[1]} & {selc_list[0]}', []))
    st.write(f'Found {len(jup)} common columns:')
    st.write(jup)

    merge_on = st.selectbox('Select what column you would love to merge these sheets on?', options= jup, index=None)

    if merge_on:
      st.markdown('##### Merged Preview:')
      df1 = pd.read_excel(file, sheets[0])
      df2 = pd.read_excel(file, sheets[1])
      merged_df = pd.merge(df1, df2, on=merge_on)
      st.dataframe(merged_df.head(10))
      st.markdown('##### Are you satisfied with your merge? If yes click the button below to see and download the whole table (if you want to)')
      full_btn = st.button('Click Here')
      if full_btn:
        st.markdown('#### Whole Table')
        st.dataframe(merged_df)
        st.session_state.data[f"{sheets[0]} & {sheets[1]} merged"] = merged_df
        
    else: st.info('Go on to select a column to merge on!')
  
  st.markdown('---')
  st.write('**Double click to go Back the view Page, else proceed to perform other tasks with merged data**')
  Back = st.button('Back')
  if Back:
    st.session_state.page = 'view_page'


