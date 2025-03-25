import streamlit as st
import pandas as pd
import numpy as np
import time
from merge import merge_func
from history import HistoryTracker
import io

if "history_tracker" not in st.session_state:
    st.session_state.history_tracker = HistoryTracker()

st.markdown("# Your Favorite Predictor App 👍")
st.sidebar.markdown("# Main page")
uploaded_file  = st.file_uploader("Upload a file to unconver insights", type=["csv", "xlsx"], key = "file_uploader")
#ser session stage for page
if 'page' not in st.session_state:
    st.session_state.page = 'view_page'

def overview_data(df):
    st.markdown("#### Featured Columns")
    st.markdown(f'**{[col for col in df.columns]}**')
    st.write({f'Number of Rows: {len(df)}'})
    st.markdown("#### Table Statistics")
    overview_dict = {'Featured Colums': [str(i) for i in df.columns],
                    'Data Type': [str(col_type) for col_type in df.dtypes.values],
                    'Column Count': [str(val) for val in df.count().values],
                    'Unique Values': [str(val) for val in df.nunique().values],
                    'Null Values Count': [str(nan) for nan in df.isnull().sum()],
                    'Percentage of Null Values': [f"{str(round((nan / df.shape[0]) * 100, 2))}%" for nan in df.isnull().sum()],
                    'Mean': df.mean(numeric_only=True).round(2).astype(str).reindex(df.columns, fill_value='N/A').tolist(),
                    'Median': df.median(numeric_only=True).round(2).astype(str).reindex(df.columns, fill_value='N/A').tolist(),
                    'Upper Quartile': df.quantile(0.75, numeric_only=True).round(2).astype(str).reindex(df.columns, fill_value= "N/A").tolist(),
                    'Lower Quartile': df.quantile(0.25, numeric_only=True).round(2).astype(str).reindex(df.columns, fill_value= "N/A").tolist(),
                    'Min': df.min(numeric_only=True).round(2).astype(str).reindex(df.columns, fill_value='N/A').tolist(),
                    'Max': df.max(numeric_only=True).round(2).astype(str).reindex(df.columns, fill_value='N/A').tolist(),
                    'Mode': df.mode().iloc[0].astype(str).reindex(df.columns, fill_value='N/A').tolist(),
                    'Variance': df.var(numeric_only=True).round(2).astype(str).reindex(df.columns, fill_value='N/A').tolist(),
                    'Standard Deviation': df.std(numeric_only=True).round(2).astype(str).reindex(df.columns, fill_value='N/A').tolist()
                }
    overview_table = pd.DataFrame(overview_dict).T
    st.dataframe(overview_table)
    st.markdown('#### Table Preview')
    st.dataframe(df.head())

@st.cache_data
def file_handler(file):
    success_prompt = st.empty()
    if file is not None:
        success_prompt.success(f"✅ {file.name} uploaded successfully.")
        time.sleep(5)
        success_prompt.empty()
        file_type = file.name.split('.')[-1]

        st.markdown("### $Desciptive$  $Overview$")
        st.write(f"File type: {file_type.upper()}")
        return file, file_type



def view(file, format):
     st.session_state.data = {}
     if format.upper() == 'CSV':
            df = pd.read_csv(file)
            overview_data(df)
            st.session_state.data[f"{file.name}"] = df
            st.session_state.history_tracker.save_version(st.session_state.data[f"{file.name}"], action=f"{file.name} Uploaded")
        
     elif format.upper() == 'XLSX':
            xlsx = pd.ExcelFile(file)
            st.session_state.excel_file = xlsx
            sheet_names = xlsx.sheet_names
            st.markdown(f"### Found {len(sheet_names)} sheets")
            st.write( sheet_names )
            selected_sheets = st.multiselect('Select what sheet you want to work with', sheet_names)

            # Preview selected sheets
            if selected_sheets:
                index ={}
                for i, sheet in enumerate(selected_sheets):
                    df = pd.read_excel(xlsx, sheet_name= sheet)
                    st.markdown(f"> ### {sheet} Descriptive Overview")
                    overview_data(df)
                    index.update({sheet : df.columns.tolist()})
                    st.session_state.data[f"{sheet}"] = df
                    st.session_state.history_tracker.save_version(st.session_state.data[f"{sheet}"], action=f"{sheet} Uploaded")

                common_columns_confirmed = {}
                keys = list(index.keys())
                for i, sheet in enumerate(keys):
                    values = index[sheet]

                    
                    for j in range(i + 1, len(keys)):  # Start from the next sheet
                        other_sheet = keys[j]
                        if sheet == other_sheet:  # If the current sheet is the same as the next sheet, skip
                             continue 
                        
                        # Find common columns between the current sheet and the next sheet
                        common_columns = set(values) & set(index[other_sheet]) 
            
                        if common_columns: 
                           st.info(f"QUICK NOTE: Common Columns were found between {sheet} and {other_sheet}: {list(common_columns)}")
                           common_columns_confirmed[f'{sheet} & {other_sheet}'] = list(common_columns)
                st.session_state.common_columns = common_columns_confirmed
                           
                # Ask the user if they want to merge the selected sheets
                if common_columns_confirmed:    
                #    st.write(common_columns_confirmed)                         
                   merge = st.radio('Will you like to merge any of these sheets?', options=['Yes', 'No'], index = None)
                   if  merge == 'Yes':
                       st.session_state.selected_sheets = selected_sheets
                       st.markdown("*Preparing to merge the selected sheets...*")
                       status = st.empty()
                       bar = st.progress(0)
                       for i in range(101):
                           status.text(f'{i}% Completed' if i!= 100 else 'Preparation Complete')
                           # Update the progress bar with each iteration.
                           bar.progress(i)
                           time.sleep(0.01)   
                       st.button("Proceed to Merge", on_click=go_to_merge)    
    
def go_to_merge():
    st.session_state.page = 'merge_page'


if st.session_state.page == 'view_page':
    if uploaded_file is not None: 
      file, format = file_handler(uploaded_file)
      view(file, format)


elif st.session_state.page == 'merge_page':
    # st.write("You are now on the merge page!")
    merge_func()

else: st.stop()

st.sidebar.markdown('-------')
st.sidebar.subheader("🔄 Data Version Control")
for i, entry in enumerate(st.session_state.history_tracker.history):
    with st.sidebar.expander(f"🔹 {entry['timestamp']} - {entry['action']}"):
        st.write(f"**Timestamp:** {entry['timestamp']}")
        st.write(f"**Action:** {entry['action']}")
          # Check if snapshot exists
        if 'data' in entry:
                snapshot_df = entry['data']  # Assuming it's a Pandas DataFrame

                # Convert to CSV for download
                csv_buffer = io.StringIO()
                snapshot_df.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()

                # Download button
                st.download_button(
                    label="⬇️ Download This Version",
                    data=csv_data,
                    file_name=f"version_{i}_{entry['timestamp']}.csv",
                    mime="text/csv"
                )

