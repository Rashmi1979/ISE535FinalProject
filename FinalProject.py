import dash
import dash_table
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import plotly.graph_objs as go
from dash import Input, Output, State, html
import matplotlib.pyplot as plt 
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


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
                html.B("Introduction: "),
                html.P("The Pima are a group of Native Americans living in Arizona. In recent years due change in life style and food habits - aregetting dignosed more with Diabetes than that of previous years. ", className="card-text"),
                html.B("Project Goals"),
                html.P("Primary goal of this project is to use various machine learning models to predict if patient has diabetes or not. Data exploration, Data Visualization and Data Modeling are essential steps to analyse, clean prepare and model the data for prediction. ", className="card-text"),
                html.B("Link to the dataset: "),
                html.P("https://www.kaggle.com/uciml/pima-indians-diabetes-database", className="card-text"),
                html.B("Dataset details: "),
                html.P("The datasets consist of several medical predictor (independent) variables and one target (dependent) variable, Outcome. Independent variables include the number of pregnancies the patient has had, their BMI, insulin level, age, and so on.", className="card-text"),
                html.Ul([
                                html.Li("Pregnancies: Number of times pregnant"),
                                html.Li("Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test"),
                                html.Li("BloodPressure: Diastolic blood pressure (mm Hg)"),
                                html.Li("SkinThickness: Triceps skin fold thickness (mm)"),
                                html.Li("Insulin: 2-Hour serum insulin (mu U/ml)"),
                                html.Li("BMI: Body mass index (weight in kg/(height in m)^2)"),
                                html.Li("DiabetesPedigreeFunction: Diabetes pedigree function"),
                                html.Li("Age: Age(years)"),
                                html.Li("Outcome: Class variable (0 or 1) 268 of 768 are 1, the others are 0"),
                    
                    
                                
                            ]) 
    
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
newDF = df.drop(columns=['Outcome'])
predictors = newDF.columns
 

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
                                options=[{'label': i, 'value': i} for i in predictors],
                                style={ 'width': '100%'},
                                labelStyle={'float': 'left', 'clear': 'left'},
                                value = ['Glucose', 'BMI']
                            )

                        ]), width=4,style={'backgroundColor':'lightgray', "padding":"10px"}
                    ),
                    dbc.Col(
                        html.Div([
                            html.Hr(),
                            html.P(id='my-div'),
                            html.Hr(),
                            html.P(id='pre-div'),
                            html.Hr(),
                            dbc.Button("Run Models", id="run_m_button", className="me-2",n_clicks=0),
                            dcc.Graph(id="graph"),
                            html.Ul([
                                html.Li("KNN: KNeighbors Classifier"),
                                html.Li("SVC: Support Vector Classfier"),
                                html.Li("LR: Logistic Regression"),
                                html.Li("DT: Decision Tree Classifier"),
                                html.Li("GNB: Gaussian NB"),
                                html.Li("RF: Random Forest Classifier"),
                                html.Li("GB: Gradient Booster Classifier"),
                                
                            ])    
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
    
    
@app.callback(
    Output(component_id='pre-div', component_property='children'),
    [Input(component_id='predictors', component_property='value')]
)
def update_predictor_div(input_value):
    return 'Selected Predictors = {}'.format(input_value)    
 
    
    
        
@app.callback(
     #Output("test-out", "children"), 
    Output('graph','figure'),
    [Input("run_m_button", "n_clicks"),
     State('my-id', 'value'),
     State('predictors','value')
    ]
)
def on_button_click(n,trainsplit,pred):
    testsplit = (100 - int(trainsplit))/100   
    df_mod = df[(df.BloodPressure != 0) & (df.BMI != 0) & (df.Glucose != 0)]
    
    X = df_mod[pred]
    y = df_mod.Outcome
    
    models = []
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('SVC', SVC()))
    models.append(('LR', LogisticRegression()))
    models.append(('DT', DecisionTreeClassifier()))
    models.append(('GNB', GaussianNB()))
    models.append(('RF', RandomForestClassifier()))
    models.append(('GB', GradientBoostingClassifier()))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= testsplit, stratify = df_mod.Outcome, random_state=0)
    names = []
    scores = []
    for name, model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scores.append(accuracy_score(y_test, y_pred))
        names.append(name)
    tr_split = pd.DataFrame({'Classifier': names, 'Score': scores})

    colors = ['lightslategray',] * 7
    highRow = tr_split['Score'].idxmax()
    colors[highRow] = 'crimson'

    
    
#    fig = px.bar(tr_split, x = 'Classifier', y = 'Score', text='Score', marker_color=colors)
    fig = go.Figure(data=[go.Bar( x = tr_split.Classifier, y = tr_split.Score,text= tr_split.Score,
    marker_color=colors # marker color can be a single color value or an iterable
)])
    
    return fig

    
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