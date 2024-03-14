import dash
from dash import callback
import statsmodels
import dash_bootstrap_components as dbc
from dash import dcc, html  # Updated imports
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import numpy as np


# Create a DataFrame
data = pd.read_csv("housing_price_dataset.csv") 
df = pd.DataFrame(data)

df.Price = df.Price.abs() # a few prices are negative (a possible Error while Data Collecting.)

# Create a Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}]
                )




# Create a dataframe for Sunburst Figure.
grouped_df = df.groupby(by=["Neighborhood", "Bedrooms", "Bathrooms"] , as_index=False).agg({"Price": "mean", })
grouped_df.Bedrooms = grouped_df.Bedrooms.apply(lambda x:  str(x) + ' Beds' )
grouped_df.Bathrooms = grouped_df.Bathrooms.apply(lambda x:  str(x) + ' Baths' )

# Create a Sunburst Figure.
sunburst_fig = px.sunburst(grouped_df, path=['Neighborhood', 'Bedrooms', 'Bathrooms'], values='Price')


# Create a Layout of our app using Dash Bootstrap Components
app.layout = dbc.Container([
    # Remove the justify attribute from this dbc row
    dbc.Row([
        dbc.Col(html.H1("House Pricing Application",
                        className="text-center text-primary mb-4"), width=12),
     
    ]),
    
    # Add a dbc row to wrap the graph and the dropdown buttons
    dbc.Row([
        # Wrap the dropdown buttons in a dbc col
        dbc.Col([
            # Dropdown section
            dbc.Row([
                dbc.Col([
                        html.Label("Select Neighborhood:", style={"font-weight": "bold"}),

                        dcc.Dropdown(
                            multi=False, value=df.Neighborhood.iloc[0],
                            options=[{'label': x, 'value': x} for x in df.Neighborhood.unique()],
                            style={"width": 180},
                            id='dropdown-neighborhood'),

                        html.Br(),

                            html.Label("Select Number of Bathrooms:", style={"font-weight": "bold"}),

                            dcc.Dropdown(multi=False, value = df.Bathrooms.min(), options=df.Bathrooms.unique(),
                                         id="dropdown-bathroom",  style={"width": 180}),

                        html.Br(),
                        html.Label("Select Number of Bedrooms:", style={"font-weight": "bold"}),
                        dcc.Dropdown(value = df.Bedrooms.min(), options=df.Bedrooms.unique(), id="dropdown-bedroom",
                                    style={"width": 180}),]),
                    ]),

        ], width=2), # Set the width of the dbc col to 6

        # Wrap the graph in a dbc col
        dbc.Col([
            dcc.Graph(id='graph-scatter'),
            dcc.RangeSlider(df.YearBuilt.min(), df.YearBuilt.max(), 10,
                    value=[df.YearBuilt.min(), df.YearBuilt.max()], id='year-slider',
                    marks= {year: str(year) for year in range(df.YearBuilt.min(), df.YearBuilt.max(), 10) }, ),
                    html.Div(id='output-container-range-slider')
                    ], 
            width={"size": 10}), ], align="center"), # Set the align attribute to "center"


    dbc.Row([

        dbc.Col([
            html.Br(),
            html.Br(),
            html.H3("Average Price per Neigborhood"),
            dcc.Graph(id='sunburst', figure=sunburst_fig),
            
        ], width=5 ,align="left"),
        dbc.Col([

                html.Br(),
                html.Br(),
                html.H1("Price Prediction"),
                html.Br(),
            html.H4("Provide the house's square footage for a price estimation:"),
        html.Br(),
        html.Br(),
        dcc.Input(id="Area-id",value=0, type="number", placeholder="Enter Area of house in SquareFeet", style={'marginRight':'180px', 'width':300}),
        html.Div(id="output-box"),
        ]),
    ])



])




@callback(
    Output('graph-scatter', 'figure'),
    
    [Input('dropdown-neighborhood', 'value'), Input('dropdown-bathroom', 'value'), 
    Input('dropdown-bedroom', 'value'), Input('year-slider', 'value'),]
)



def update_graph(value1, value2, value3, value4):


    dff = df[(df.Neighborhood==value1) & (df.Bathrooms==value2) & (df.Bedrooms==value3) & (df.YearBuilt.between( value4[0], value4[1] )) ]

    scatter_fig = px.scatter(dff, x='SquareFeet', y='Price', color="Neighborhood", trendline="ols", trendline_color_override="red")
    
                              
    return scatter_fig
    


@callback(
    Output('output-box', 'children'),
    [Input('dropdown-neighborhood', 'value'), Input('dropdown-bathroom', 'value'), 
    Input('dropdown-bedroom', 'value'), Input('year-slider', 'value'),Input('Area-id', 'value')]
     
)

def price_recommendation(value1, value2, value3, value4, value5):
   
    # select the rows with given conditions
    data_cluster = df[(df.Neighborhood==value1) &
            (df.Bathrooms==value2) &  
            (df.Bedrooms==value3) & 
            (df.YearBuilt.between( value4[0], value4[1] )) ]


    # create a linear regression 
    a, b = np.polyfit(x=data_cluster.SquareFeet, y=data_cluster.Price, deg=1)

    if value5 != 0:
        fairestـprice = round(a,2)*(value5) + round(b, 2)


        # calculate a 
        mean = data_cluster.Price.mean()
        std_dev = data_cluster.Price.std()
        
        std_error = data_cluster.Price.std()/np.sqrt(len(data_cluster.Price))
        
        value_x = fairestـprice
        
        z_score = (value_x - mean) / std_dev
        
        lower_limit = fairestـprice - (z_score*std_error)
        upper_limit = fairestـprice + (z_score*std_error)

        return f"""f(x) = {round(a,2)}x + {round(b, 2)}
                for a {value5} SquareFeet house with {value2} bathrooms and {value3} bedrooms and built in years {value4[0]} - {value4[1]} at {value1} 
                Based on linear regression model,
                fairest price is: {fairestـprice}, 
                lowest price is: {lower_limit},
                highest price is: {upper_limit}"""














if __name__ == '__main__':
    app.run_server(debug=True)
