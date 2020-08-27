# Import general libraries
import pandas as pd
import numpy as np
import pprint
import datetime
import random
import os
import regex as re

# Import dash
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import plotly
import plotly.express as px

app = dash.Dash(external_stylesheets=[dbc.themes.CERULEAN])

app.title = "DSTA Smart Search System"

app_name = "DSTA Smart Search System"

server = app.server  # for deployment


# Navigation bar
navbar = dbc.Navbar(
    [
        html.A(
            dbc.Row(
                [
                    dbc.Col(
                        dbc.NavbarBrand(
                            "DSTA Smart Search System",
                            className="ml-auto",
                            style={"font-size": 30},
                        )
                    ),
                ],
                align="left",
                no_gutters=True,
            ),
            href="",  # to be updated
        ),
        dbc.NavbarToggler(id="navbar-toggler"),
    ],
    color="dark",
    dark=True,
    style={"width": "100%"},
)

searchTab = html.Div(
    children=[
        html.H3("Please input your query"),
        # Input box
        dbc.Textarea(
            id="text_search",
            className="mb-3",
            placeholder="e.g. What is computational complexity principle?",
        ),
        html.Button("Search", id="button_search"),
        html.Br(),
        html.Hr(style={"height": "12", "background": "black"}),
        html.H3("Results"),
        # Output table
        html.Div(
            id="output_search_table",
            style={
                "marginTop": -20,
                "marginLeft": 10,
                "width": "80%",
                # "display": "inline-block",
            },
        ),
    ],
    style={
        "marginLeft": 40,
        "width": "80%",
        # "display": "inline-block",
    },
)

faqTab = html.Div(
    children=[
        html.H3("Top 10 Questions"),
        html.Div(
            html.H3("Work in progress"),
            style={
                "marginTop": -20,
                "marginLeft": 10,
                "width": "80%",
                # "display": "inline-block",
            },
        ),
    ]
)

# Layout of entire app
app.layout = html.Div(
    [
        navbar,
        dbc.Tabs(
            [
                dbc.Tab(searchTab, id="label_tab1", label="Smart Search"),
                dbc.Tab(faqTab, id="label_tab2", label="FAQ"),
            ],
            style={"font-size": 20, "background-color": "#b9d9eb"},
        ),
    ]
)

##### For searchTab
@app.callback(
    Output("output_search_table", "children"),  # output results in table
    [Input("button_search", "n_clicks")],  # upon clicking search button
    [State("text_search", "value")],
)  # retrieve query
def search(n_clicks, text_search):
    # model = bla
    # pred = bla
    df = pd.DataFrame({"Results": [text_search] * 10})
    output = dash_table.DataTable(
        id="dash_tb",
        columns=[{"name": "Results", "id": "Results"}],
        data=df.to_dict("records"),
        style_header={"display": "none"},
        style_data={"whiteSpace": "normal", "height": "auto", "textAlign": "left",},
    )

    return output


##### For faqTab
# @app.callback(
#     Output("output_faq_table", "children"),  # output results in table
#     # [Input("button_search", "n_clicks")],  # upon clicking search button
#     # [State("text_search", "value")],
# )  # retrieve query
def faq():
    # model = bla
    # pred = bla
    df = pd.DataFrame({"Results": ["bla bla bla"] * 10})
    output = dash_table.DataTable(
        id="dash_tb",
        columns=[{"name": "Results", "id": "Results"}],
        data=df.to_dict("records"),
        style_header={"display": "none"},
        style_data={"whiteSpace": "normal", "height": "auto", "textAlign": "left",},
    )

    return output


if __name__ == "__main__":
    app.run_server(debug=False)  # change debug to true when developing
