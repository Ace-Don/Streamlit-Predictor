
import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px

st.markdown('# See What Your Data Says! 📈📊')
plotly_visuals = ["Pivot Table","Scatter Plot", "Line Chart", "Bar Chart", "Histogram","Box Plot", "Pie Chart","Heatmap", "Bubble Chart","Area Chart", "Violin Plot", "3D Scatter Plot", "Sunburst Chart"]
def visualize_data(datasets):
    if datasets is not None:
       st.markdown(f'##### The following tables have been stored for you for use:')
       for i,data in enumerate(datasets.keys()):
           st.markdown(f'###### {i+1}. {data}  \n')   
       selected = st.selectbox('Choose a table to visualize:', options=list(datasets.keys()), index=None)   
       if selected:
        #    status = st.empty()
        #    bar = st.progress(0)

        #    ############################################
        #    for i in range(101):
        #        status.markdown(f'*Collecting Tables*' if i!= 100 else '*Collection Complete*')
        #        # Update the progress bar with each iteration.
        #        bar.progress(i)
        #        time.sleep(0.01)  
        #    time.sleep(2)    
        #    bar.empty() 
        #    status.empty()
        #    ############################################

           # Check if the selected column is numerical and binary (e.g., 0/1 values)
           df = datasets[selected]
           for x in df.columns:
               if df[x].dtype in ['int64', 'float64'] and df[x].nunique() <= 5:
                # Convert binary numerical column to categorical
                df[x] = pd.Categorical(df[x])

           st.markdown('---')
           visual = st.selectbox('Choose a visualization type:', options=plotly_visuals, index=None)

           #Filter Visual
           if visual == 'Scatter Plot':
               x = st.sidebar.selectbox('Plot', options=df.columns, index = None)
               y = st.sidebar.selectbox('Against', options=df.columns, index = None)
               legend = st.sidebar.selectbox('Legend (optional)', options=df.columns, index = None)
               if x and y or legend:
                   fig = px.scatter(df, x=x, y=y, color=legend , title = f'{x} vs {y} with {legend} as Color' if x and y and legend  
                                                                                            else f'Scatter Plot of {x} vs {y}')
                   st.plotly_chart(fig)
            
           elif visual == 'Pivot Table':
               index_col = st.sidebar.multiselect("Row Index(es)", df.columns, default=None)
               column_col = st.sidebar.multiselect("Column Index(es)", df.columns, default=None)
               value_col = st.sidebar.multiselect("Value(s)", df.columns, default=None)
               agg_func = st.sidebar.selectbox("Aggregation Function", ["sum", "mean", "count", "min", 
                                                                        "max", "median", "std", "var", 
                                                                        "prod", "first", "last"], index=None)

               if (index_col or column_col or value_col) and agg_func:
                   pivot_table = df.pivot_table(index=index_col, columns=column_col, values=value_col, aggfunc=agg_func)
                   st.subheader("Pivot Table")
                   st.dataframe(pivot_table)

               else:st.info('Select at least an index, value and one aggregation function')

           elif visual == 'Box Plot':
                x = st.sidebar.selectbox('Plot', options=df.columns, index = None)
                y = st.sidebar.selectbox('Against (optional)', options=df.columns, index = None)
                legend = st.sidebar.selectbox('Legend (optional)', options=df.columns, index = None)
                if x or y or legend:
                    fig = px.box(df, y=y, x=x, color=legend, title =  f'Box Plot of {x} vs {y} with {legend} as Color' if x and y and legend  
                                                                                        else f'Box Plot of {x} vs {y}' if x and y 
                                                                                        else f'Box Plot of {y} with {legend} as Color' if y and legend 
                                                                                        else f'Box Plot of {x} with {legend} as Color' if x and legend 
                                                                                        else f'Box Plot of {x}')
                    st.plotly_chart(fig)

           elif visual == 'Bar Chart':   
                x = st.sidebar.selectbox('Plot', options=df.columns, index = None)
                legend = st.sidebar.selectbox('Legend (optional)', options=df.columns, index = None)
                if x  or legend:
                    fig = px.bar(df, x=x, color=legend, title =  f'Count of {x} with {legend} as Color' if x and legend  
                                                                                        else f'Count of {x}')
                    st.plotly_chart(fig)

           elif visual == 'Histogram':
                x = st.sidebar.selectbox('Plot', options=df.columns, index = None)
                legend = st.sidebar.selectbox('Legend (optional)', options=df.columns, index = None)
                if x  or legend:
                    fig = px.histogram(df, x=x ,color=legend, nbins=50 ,title =  f'Histogram of {x} with {legend} as Color' if x and legend  
                                                                                        else f'Histogram of {x}')
                    
                    fig.update_layout(
                        xaxis_title=f"{x}",  # X-axis label
                        yaxis_title="Count",  # Y-axis label
                        bargap= 0.1,  # Space between bars
                        plot_bgcolor= None,  # Background color of the plot area
                        hovermode="x unified",  # Hovering behavior across the bars
                        showlegend=True  # Show legend for color differentiation
                    )

                    # Update the x-axis to make it more readable
                    fig.update_xaxes(
                        showgrid=True,  # Show gridlines on the x-axis
                        zeroline=False  # Don't show the zero line
                    )

                    # Update the y-axis for better visibility
                    fig.update_yaxes(
                        showgrid=True,  # Show gridlines on the y-axis
                        zeroline=False  # Don't show the zero line
                    )
                    st.plotly_chart(fig) 

           elif visual == 'Line Chart':
                x = st.sidebar.selectbox('Plot', options=df.columns, index = None)
                y = st.sidebar.selectbox('Against', options=df.columns, index = None)
                legend = st.sidebar.selectbox('Legend (optional)', options=df.columns, index = None)
                if x and y or legend:
                    fig = px.line(df, x=x, y=y, color=legend, title =  f'{x} vs {y} with {legend} as Color' if x and y and legend  
                                                                                        else f'Line Chart of {x} vs {y}')
                    st.plotly_chart(fig)

           elif visual == 'Pie Chart':
                x = st.sidebar.selectbox('Data', options=df.columns, index = None)
                if x:
                    count = df[x].value_counts()
                    fig = px.pie(count, names = count.index, values=count.values, hole= 0.2, title=f'Pie Chart of {x}')
                    st.plotly_chart(fig)
                          
           elif visual == 'Bubble Chart':
                x = st.sidebar.selectbox('X-Axis', options=df.columns, index = None)
                y = st.sidebar.selectbox('Y-Axis', options=df.columns, index = None)
                size = st.sidebar.selectbox('Size', options=df.columns, index = None)
                legend = st.sidebar.selectbox('Legend (Optional)', options=df.columns, index = None)
                if x and y and size or legend:
                    fig = px.scatter(df, x=x, y=y, size=size, color=legend, title =  f'Bubble Chart of {x} vs {y} with Size as {size} and Color as {legend}')
                    st.plotly_chart(fig)

           elif visual == 'Sunburst Chart':
                x = st.sidebar.multiselect('Hierachy', options=df.columns, default = [])
                if x:
                    fig = px.sunburst(df, path=x, color_continuous_scale="Viridis", title=f'Sunburst Chart with Hierarchy {", ".join(x)}')
                    st.plotly_chart(fig)


           elif visual == 'Violin Plot':
                y = st.sidebar.selectbox('Plot', options=df.columns, index = None)
                x = st.sidebar.selectbox('Against (optional)', options=df.columns, index = None)
                legend = st.sidebar.selectbox('Legend (optional)', options=df.columns, index = None)
                if x or y or legend:
                    fig = px.violin(df, y=y, x=x, color=legend, box=True, points="all", title =  f'Violin Plot of {x} vs {y} with {legend} as Color' if x and y and legend  
                                                                                        else f'Violin Plot of {x} vs {y}' if x and y 
                                                                                        else f'Violin Plot of {y} with {legend} as Color' if y and legend
                                                                                        else f'Violin Plot of {x} with {legend} as Color' if x and legend
                                                                                        else f'Violin Plot of {x}' if x
                                                                                        else f'Violin Plot of {y}')
                    
                    fig.update_layout(
                        width=1000,  
                        height=600,
                    )

                    st.plotly_chart(fig)  
                       
           elif visual == 'Area Chart':
                x = st.sidebar.selectbox('X-Axis', options=df.columns, index = None)
                y = st.sidebar.selectbox('Y-Axis', options=df.columns, index = None)
                legend = st.sidebar.selectbox('Color (Optional)', options=df.columns, index = None)
                if x and y or legend:
                    fig = px.area(df, x=x, y=y, color=legend, title =  f'Area Chart of {x} vs {y} with Color as {legend}' if x and y and legend
                                                                        else f'Area Chart of {x} vs {y}')
                    
                    fig.update_layout(
                        width=1000,  
                        height=570,
                    )
                    
                    st.plotly_chart(fig)

           elif visual == 'Heatmap':
                x = st.sidebar.selectbox('Plot', options=df.columns, index = None)
                y = st.sidebar.selectbox('Against', options=df.columns, index = None)
                if x and y :
                    fig = px.density_heatmap(df, x=x, y=y, title=f'Heatmap of {x} vs {y}')
                    st.plotly_chart(fig)   

           elif visual == "3D Scatter Plot":
                x = st.sidebar.selectbox('X-Axis', options=df.columns, index = None)
                y = st.sidebar.selectbox('Y-Axis', options=df.columns, index = None)
                z = st.sidebar.selectbox('Z-Axis', options=df.columns, index = None)
                legend = st.sidebar.selectbox('Color (Optional)', options=df.columns, index = None)
                if x and y and z:
                    fig = px.scatter_3d(df, x=x, y=y, z=z, color= legend, title=f'3D Scatter Plot of {x}, {y}, and {z}')
                    st.plotly_chart(fig)     

                fig.update_layout(
                        width=1000,  
                        height=670,
                    )          


if "data" not in st.session_state:
    st.info('Upload data on the main page.') 

else:    
    visualize_data(st.session_state.data)   
