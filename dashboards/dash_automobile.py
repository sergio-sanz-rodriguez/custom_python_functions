#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import dash
import more_itertools
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px

# Load the data using pandas
data = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/historical_automobile_sales.csv')

# Initialize the Dash app
app = dash.Dash(__name__)

# Set the title of the dashboard
app.title = "Automobile Statistics Dashboard"

#---------------------------------------------------------------------------------
# Create the dropdown menu options
dropdown_options = [
    {'label': 'Yearly Statistics', 'value': 'Yearly Statistics'},
    {'label': 'Recession Period Statistics', 'value': 'Recession Period Statistics'}
]
# List of years 
year_list = [i for i in range(1980, 2024, 1)]
#---------------------------------------------------------------------------------------
# Create the layout of the app
app.layout = html.Div([
    #TASK 2.1 Add title to the dashboard
    html.H1('Automobile Statistics Dashboard', style={'textAlign': 'center', 'color': '#503D36', 'font-size': 24}),
    html.Div([#TASK 2.2: Add two dropdown menus
        html.Label("Select Statistics:", style={'font-size':'20px'}),
        dcc.Dropdown(
            id='dropdown-statistics',
            options=dropdown_options,
            value='Select Statistics',
            placeholder='Select a report type',
            style={'width': '40%', 'font-size':'20px'} 
            #, 'padding': '3px', 'font-size':'20px', 'textAlign': 'center'}
        )
    ]),
    html.Div(dcc.Dropdown(
            id='select-year',
            options=[{'label': i, 'value': i} for i in year_list],
            value='2010',
            placeholder='Select a year',
            style={'width': '40%', 'font-size':'20px'} 
        )),
    html.Div([#TASK 2.3: Add a division for output display
    html.Div(id='output-container', className='char-grid', style={'display': 'flex'}),])
])
#TASK 2.4: Creating Callbacks
# Define the callback function to update the input container based on the selected statistics
@app.callback(
    Output(component_id='select-year', component_property='disabled'),
    Input(component_id='dropdown-statistics',component_property='value'))

def update_input_container(selected_statistics):
    if selected_statistics =='Yearly Statistics': 
        return False
    else: 
        return True

#Callback for plotting
# Define the callback function to update the input container based on the selected statistics
@app.callback(
    Output(component_id='output-container', component_property='children'),
    [Input(component_id='select-year', component_property='value'), Input(component_id='dropdown-statistics', component_property='value')])


def update_output_container(input_year, selected_statistics):
    if selected_statistics == 'Recession Period Statistics':
        # Filter the data for recession periods
        recession_data = data[data['Recession'] == 1]


    
        
#TASK 2.5: Create and display graphs for Recession Report Statistics

#Plot 1 Automobile sales fluctuate over Recession Period (year wise)
        # use groupby to create relevant data for plotting
        yearly_rec = recession_data.groupby('Year')['Automobile_Sales'].mean().reset_index()
        fig1 = px.line(yearly_rec, 
                x='Year',
                y='Automobile_Sales',
                title="Average Automobile Sales fluctuation over Recession Period", labels={'Automobile_Sales':'Automobile Sales'})
        fig1.update_layout(title={'x': 0.5, 'xanchor': 'center'})
        R_chart1 = dcc.Graph(figure=fig1)

#Plot 2 Calculate the average number of vehicles sold by vehicle type       
        # use groupby to create relevant data for plotting
         #df.groupby(['Recession','Vehicle_Type'])['Automobile_Sales'].mean().reset_index()
        average_sales = recession_data.groupby('Vehicle_Type')['Automobile_Sales'].mean().reset_index()
        fig2 = px.bar(average_sales,
                x='Vehicle_Type',
                y='Automobile_Sales',
                title="Average Number of Vehicles Sold by Vehicle Type",
                labels={'Vehicle_Type':'Vehicle Type', 'Automobile_Sales':'Automobile Sales'})
        fig2.update_layout(title={'x': 0.5, 'xanchor': 'center'})
        R_chart2  = dcc.Graph(figure=fig2)
        
# Plot 3 Pie chart for total expenditure share by vehicle type during recessions
        # use groupby to create relevant data for plotting
        exp_rec= recession_data.groupby('Vehicle_Type')['Advertising_Expenditure'].sum().reset_index()
        fig3=px.pie(exp_rec,
                values='Advertising_Expenditure',
                names='Vehicle_Type',
                title='Total Expenditure Share by Vehicle Type',
                labels={'Advertising_Expenditure':'Advertising Expenditure', 'Vehicle_Type':'Vehicle Type'})
        fig3.update_layout(title={'x': 0.5, 'xanchor': 'center'})
        R_chart3 = dcc.Graph(figure=fig3)


# Plot 4 bar chart for the effect of unemployment rate on vehicle type and sales
        #unemployment_rec = recession_data.sort_values(by="unemployment_rate")
        unemployment_rec = recession_data.groupby('Vehicle_Type')[['unemployment_rate','Automobile_Sales']].mean().reset_index()
        fig4 = px.bar(unemployment_rec,
                x='Vehicle_Type',
                y='Automobile_Sales',
                color='unemployment_rate',                
                title="Effect of Unemployment Rate on Vehicle Type and Sales",
                labels={'unemployment_rate':'Unemployment Rate', 'Automobile_Sales':'Automobile Sales'})
        fig4.update_layout(title={'x': 0.5, 'xanchor': 'center'})
        R_chart4 = dcc.Graph(figure=fig4)


        return [
            html.Div(className='chart-item', children=[html.Div(children=R_chart1),html.Div(children=R_chart2)]),
            html.Div(className='chart-item', children=[html.Div(children=R_chart3),html.Div(children=R_chart4)])
            ]

# TASK 2.6: Create and display graphs for Yearly Report Statistics
# Yearly Statistic Report Plots                             
    elif (input_year and selected_statistics=='Yearly Statistics') :
        yearly_data = data[data['Year'] == input_year]
                              
#TASK 2.5: Creating Graphs Yearly data
                              
#plot 1 Yearly Automobile sales using line chart for the whole period.
        yas= data.groupby('Year')['Automobile_Sales'].mean().reset_index()
        fig1 = px.line(yas,
                x='Year',
                y='Automobile_Sales',
                title="Yearly Automobile Sales",
                labels={'Automobile_Sales':'Automobile Sales'})
        fig1.update_layout(title={'x': 0.5, 'xanchor': 'center'})        
        Y_chart1 = dcc.Graph(figure=fig1)
            
# Plot 2 Total Monthly Automobile sales using line chart.
        monthly_sales = yearly_data.groupby('Month')['Automobile_Sales'].sum().reset_index()
        fig2 = px.line(yearly_data,
                x='Month',
                y='Automobile_Sales',
                title="Total Monthly Automobile Sales in the Year {}".format(input_year),
                labels={'Automobile_Sales':'Automobile Sales'})
        fig2.update_layout(title={'x': 0.5, 'xanchor': 'center'})
        Y_chart2 = dcc.Graph(figure=fig2)

            # Plot bar chart for average number of vehicles sold during the given year
        avr_vdata=yearly_data.groupby('Vehicle_Type')['Automobile_Sales'].mean().reset_index()
        fig3 = px.bar(avr_vdata,
                x='Vehicle_Type',
                y='Automobile_Sales',
                title='Average Vehicles Sold by Vehicle Type in the Year {}'.format(input_year),
                labels={'Automobile_Sales':'Automobile Sales', 'Vehicle_Type':'Vehicle Type'})
        fig3.update_layout(title={'x': 0.5, 'xanchor': 'center'})
        Y_chart3 = dcc.Graph(figure=fig3)

            # Total Advertisement Expenditure for each vehicle using pie chart
        exp_year = yearly_data.groupby('Vehicle_Type')['Advertising_Expenditure'].sum().reset_index()
        fig4=px.pie(exp_year,
                values='Advertising_Expenditure',
                names='Vehicle_Type',
                title='Total Expenditure Share by Vehicle Type in the Year {}'.format(input_year),
                labels={'Advertising_Expenditure':'Advertising Expenditure', 'Vehicle_Type':'Vehicle Type'})
        fig4.update_layout(title={'x': 0.5, 'xanchor': 'center'})
        Y_chart4 = dcc.Graph(figure=fig4)

#TASK 2.6: Returning the graphs for displaying Yearly data
        return [
                html.Div(className='chart-item', children=[html.Div(children=Y_chart1),html.Div(children=Y_chart3)]),
                html.Div(className='chart-item', children=[html.Div(children=Y_chart2),html.Div(children=Y_chart4)])
                ]
        
    else:
        return None

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)

