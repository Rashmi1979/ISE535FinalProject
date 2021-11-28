import dash
import dash_table
import pandas as pd
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import plotly.graph_objs as go
from dash import Input, Output, State, html
import matplotlib.pyplot as plt 
import plotly.express as px


app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP],suppress_callback_exceptions=True)

nav_item1 = dbc.NavItem(dbc.NavLink("About", href="/about"))
nav_item2 = dbc.NavItem(dbc.NavLink("Data Exploration", href="/explore"))
nav_item3 = dbc.NavItem(dbc.NavLink("Data Visualization", href="/visual"))
nav_item4 = dbc.NavItem(dbc.NavLink("Data Modeling", href="/model"))

navbar = dbc.Navbar(
    dbc.Container(
        [   dbc.NavbarBrand("PIMA Indian Diabetes Data Anlaysis", href="#"),
            dbc.Nav([nav_item1,nav_item2,nav_item3, nav_item4], className="ms-auto", navbar=True),
        ]
    ), className="mb-5",sticky="top",
)

aboutPage = html.Div([
                html.H4("About this Project", className="card-title"),
                html.P("Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum", className="card-text",
                )
            ]
            ,
            style={"width": "100%"},
    )

# Read the file
df = pd.read_csv('diabetes.csv')

PAGE_SIZE = 10
explorePage =  html.Div([
            html.H4("Data Exploration ", className="card-title"),
            html.P("Below data table supports following features: "),
            html.Ul([
                html.Li("Sort values for any column, it support multi column sort as well"),
                html.Li("Filter data by providing condition for required column/s"),
                html.Li("Delete columns or rows"),
                html.Li("Edit cell value"),
                html.Li("Default record size per page and page navigation"),
                html.Li("Export filtered to CSV using Export button"),
            ]),
    
            dash_table.DataTable(
                id='datatable-interactivity',
                columns=[ {"name": i, "id": i, "deletable": True, "selectable": True} for i in df.columns],
                data=df.to_dict('records'),
                editable=True,
                filter_action="native",
                sort_action="native",
                sort_mode="multi",
                column_selectable="single",
                row_selectable="multi",
                row_deletable=True,
                selected_columns=[],
                selected_rows=[],
                page_action="native",
                page_current= 0,
                page_size= PAGE_SIZE,
                export_format="csv"
            ),
            html.Div(id='datatable-interactivity-container'),
        ],
        style={"width": "100%"},
)

@app.callback(
    Output('datatable-interactivity', 'style_data_conditional'),
    Input('datatable-interactivity', 'selected_columns')
)
def update_styles(selected_columns):
    return [{
        'if': { 'column_id': i },
        'background_color': '#D2F3FF'
    } for i in selected_columns]


############   VISUAL PAGE STARTING HERE ##############


corr = df.corr()    
columnValues = df.columns  
df['Outcome'] = df['Outcome'].astype('category')

visualPage =  html.Div([
            html.H4("Data Visualization ", className="card-title"),
            html.P("..... summaries below... "),
    
            html.H5("Correlation Matrix"),
            dcc.Graph(
                id='heatMap',
                figure={
                    'data' : [
                        go.Heatmap(
                            z=corr.values,
                            x=corr.columns.tolist(),
                            y=corr.index.tolist(),
                            colorscale='blues'
                        )
                    ],
                    'layout' : go.Layout(
                        title = "Data correlation matrix",
                        xaxis = dict(ticks=''),
                        yaxis = dict(ticks='',automargin=True)
                    )
                }
            ),
            html.Hr(),
            html.Div([
                html.H5("Histograms"),
                dcc.Dropdown(
                    id='xaxis',
                    options=[{'label': i.title(), 'value': i} for i in columnValues],
                    value=columnValues[0],
                )
            ], style={'width': '48%'}),
            
            dcc.Graph(id='histograms'),
            html.Hr(),
            html.Div([
                html.H5("Scattered Plots"),
                dbc.Row(
                [
                    dbc.Col("X Axis"),
                    dbc.Col("Y Axis"),
                ]),
   
                dbc.Row(
                [
                    dbc.Col( dcc.Dropdown(
                                id='xaxis-column',
                                options=[{'label': i, 'value': i} for i in columnValues],
                                value=columnValues[0]
                             ),
                    ),
                    dbc.Col(dcc.Dropdown(
                                id='yaxis-column',
                                options=[{'label': i, 'value': i} for i in columnValues],
                                value=columnValues[1]
                            ),
                    ),
                ]),
   
            ]),
    
            dcc.Graph(id='scattered-plots'),
     
    ]
)

@app.callback(
    Output('histograms', 'figure'),
    [Input('xaxis', 'value')])
def update_graph(xaxis_name):
    return  px.histogram(df,
            x = xaxis_name,
            color="Outcome")
    
@app.callback(
    Output('scattered-plots', 'figure'),
    [Input('xaxis-column', 'value'),
     Input('yaxis-column', 'value')])
def update_sc_graph(xaxis_name, yaxis_name):
    return px.scatter(df,
                x= xaxis_name,
                y= yaxis_name,
                color="Outcome",
                color_continuous_scale = px.colors.sequential.Inferno)
      
    
############   MODEL PAGE STARTING HERE ##############

#df = df[df.BMI==0]
#median_bmi = df['BMI'].median()
#df['BMI'] = df['BMI'].replace(to_replace=0, value=median_bmi)

#mBP = df['BloodPressure'].median()
#df['BloodPressure'] = df['BloodPressure'].replace(to_replace=0, value=mBP)

#mSkin = df['SkinThickness'].median()
#df['SkinThickness'] = df['SkinThickness'].replace(to_replace=0, value=mSkin)

#mGlucose = df['Glucose'].median()
#df['Glucose'] = df['Glucose'].replace(to_replace=0, value=mGlucose)


#################


    
modelPage = html.Div([
            html.H4("Data Modeling", className="card-title"),
            html.P(".....  below... "),
            dbc.Row(
                [
                    dbc.Col( 
                        html.Div([
                             html.B("Select % for training dataset:"),
                             dcc.Input(id='my-id', value='70', type='text'),
                             html.Hr(),
                             html.B("Select predictors from list below:"),
                             dcc.Checklist(
                                id='predictors',
                                options=[{'label': i, 'value': i} for i in columnValues],
                                style={ 'width': '100%'},
                                labelStyle={'float': 'left', 'clear': 'left'}
                            )

                        ]), width=4,style={'backgroundColor':'lightgray', "padding":"10px"}
                    ),
                    dbc.Col(
                        html.Div([
                           html.Span(id='my-div')
                        ]), width=8  
                    ),
                ]),
    
        ]
)
    
@app.callback(
    Output(component_id='my-div', component_property='children'),
    [Input(component_id='my-id', component_property='value')]
)
def update_output_div(input_value):
    return 'Train Dataset % = {}'.format(input_value)    
    
    
    
    
app.layout = html.Div(
    [
        dcc.Location(id="url", pathname="/about"), navbar,
        dbc.Container(id="content", style={"padding": "20px"}),
    ]
)

@app.callback(Output("content", "children"), [Input("url", "pathname")])
def display_page(pathname):
    if pathname == "/about":
        return aboutPage
    if pathname == "/explore":
        return explorePage
    if pathname == "/visual":
        return visualPage
    if pathname == "/model":
        return modelPage
    # if not recognised, return 404 message
    return html.P("404 - page not found")



if __name__ == "__main__":
    app.run_server(debug=True)