from dash import html, dcc
import pandas as pd


def create_layout(raw_distros=None,
                  preprocessed_distros=None,
                  nan_distro=None,
                  feature_distro=None,
                  age_distro=None,
                  verbose =False):
    if verbose: 
        print("Received figure type:", type(raw_distros[0]))
    if feature_distro is not None:
        num_columns = feature_distro.columns
        
    return html.Div([
        html.H1("Titanic Spaceship Data Analysis"),
        # Tabs for different plots
        dcc.Tabs([
            dcc.Tab(label="Feature-Feature Analysis", children=[
            #html.H1("Feature-Feature Scatter Plot"),
            # Dropdowns to select X and Y features
            html.Label("Select X-axis Feature:"),
            dcc.Dropdown(
                id='x-feature',
                options=[{'label': col, 'value': col} for col in num_columns],
                value=num_columns[0]  # Default selection
            ),
            html.Label("Select Y-axis Feature:"),
            dcc.Dropdown(
                id='y-feature',
                options=[{'label': col, 'value': col} for col in num_columns],
                value=num_columns[1]  # Default selection
            ),
            # Scatter plot
            dcc.Graph(id='feature-scatter'),
            ]),
            dcc.Tab(label="Features", children=[
                dcc.Graph(figure=age_distro)
            ]),
            dcc.Tab(label="Inspecting the raw data", children=[
                *[dcc.Graph(figure=distro) for distro in raw_distros]
            ]),
            dcc.Tab(label="Inspecting Cleaned Data", children=[
                *[dcc.Graph(figure=distro) for distro in preprocessed_distros]
            ]),
            dcc.Tab(label="NAN's Distribution", children=[
                *[dcc.Graph(figure=distro) for distro in nan_distro]
            ]), 
        ], id = "my-tabs")
    ])