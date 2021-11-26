
import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, html

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

# make a reuseable navitem for the different examples
nav_item1 = dbc.NavItem(dbc.NavLink("About", href="#about"))
nav_item2 = dbc.NavItem(dbc.NavLink("Data Visualization", href="#visual"))
nav_item3 = dbc.NavItem(dbc.NavLink("Data Modeling", href="#model"))


# here's how you can recreate the same thing using Navbar
# (see also required callback at the end of the file)
custom_default = dbc.Navbar(
    dbc.Container(
        [
            dbc.NavbarBrand("PIMA Indian Diabetes Data Anlaysis", href="#"),
            dbc.Nav(
                    [nav_item1,nav_item2,nav_item3], className="ms-auto", navbar=True
            ),
        ]
    ),
    className="mb-5",
)


app.layout = html.Div(
    [custom_default]
)


if __name__ == "__main__":
    app.run_server(debug=True)