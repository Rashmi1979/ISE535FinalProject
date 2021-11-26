import dash
import dash_table
import pandas as pd
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dash import Input, Output, State, html

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP],suppress_callback_exceptions=True)

# make a reuseable navitem for the different examples
nav_item1 = dbc.NavItem(dbc.NavLink("About", href="/about"))
nav_item2 = dbc.NavItem(dbc.NavLink("Data Visualization", href="/visual"))
nav_item3 = dbc.NavItem(dbc.NavLink("Data Modeling", href="/model"))


# here's how you can recreate the same thing using Navbar
# (see also required callback at the end of the file)
navbar = dbc.Navbar(
    dbc.Container(
        [   dbc.NavbarBrand("PIMA Indian Diabetes Data Anlaysis", href="#"),
            dbc.Nav([nav_item1,nav_item2,nav_item3], className="ms-auto", navbar=True),
        ]
    ), className="mb-5",sticky="top",
)

aboutPage = html.Div([
                html.H4("About this Project", className="card-title"),
                html.P("Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum",
                    className="card-text",
                )
            ]
            ,
            style={"width": "100%"},
    )

# Read the file
# Left bar
# Get Dropdown for filter
# Get all columns
# Right side display table with pagination
df = pd.read_csv('diabetes.csv')

PAGE_SIZE = 10

visualizationPage =  html.Div([
            html.H4("Data Visualization ", className="card-title"),
            dash_table.DataTable(
                id='table-multicol-sorting',
                columns=[
                    {"name": i, "id": i} for i in (df.columns)
                ],
                page_current=0,
                page_size=PAGE_SIZE,
                page_action='custom',

                sort_action='custom',
                sort_mode='multi',
                sort_by=[]
            ),
        ],
        style={"width": "100%"},
)

@app.callback(
    Output('table-multicol-sorting', "data"),
    Input('table-multicol-sorting', "page_current"),
    Input('table-multicol-sorting', "page_size"),
    Input('table-multicol-sorting', "sort_by"))
def update_table(page_current, page_size, sort_by):
    print(sort_by)
    if len(sort_by):
        dff = df.sort_values(
            [col['column_id'] for col in sort_by],
            ascending=[
                col['direction'] == 'asc'
                for col in sort_by
            ],
            inplace=False
        )
    else:
        # No sort is applied
        dff = df

    return dff.iloc[
        page_current*page_size:(page_current+ 1)*page_size
    ].to_dict('records')
    

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
    if pathname == "/visual":
        return visualizationPage
    # if not recognised, return 404 message
    return html.P("404 - page not found")



if __name__ == "__main__":
    app.run_server(debug=True)