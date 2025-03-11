from dash import html, dcc
import pandas as pd


def create_layout(raw_distros=None,
                  preprocessed_distros=None,
                  nan_distro=None,
                  verbose =False):
    if verbose: 
        print("Received figure type:", type(raw_distros[0]))
    return html.Div([
        html.H1("Titanic Spaceship Data Analysis"),
        # Tabs for different plots
        dcc.Tabs([
            dcc.Tab(label="Inspecting the raw data", children=[
                *[dcc.Graph(figure=distro) for distro in raw_distros]
            ]),
            dcc.Tab(label="Inspecting Cleaned Data", children=[
                *[dcc.Graph(figure=distro) for distro in preprocessed_distros]
            ]),
            dcc.Tab(label="NAN's Distribution", children=[
                *[dcc.Graph(figure=distro) for distro in nan_distro]
            ]),
        ])
    ])