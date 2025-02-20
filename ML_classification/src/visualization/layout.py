from dash import html, dcc
import pandas as pd


def create_layout(family_distro,
                  homeplanet_distro,
                  verbose =False):
    if verbose: 
        print("Received figure type:", type(family_distro))
    return html.Div([
        html.H1("Inspecting Passengers data"),
        # Tabs for different plots
        dcc.Tabs([
            dcc.Tab(label="Inspecting Raw Data", children=[
                dcc.Graph(figure=family_distro) ,
            
                dcc.Graph(figure=homeplanet_distro)  # Second Tab
            ]),
            dcc.Tab(label="Inspecting Cleaned Data", children=[
                html.H1("Inspecting Cleaned Data")
            ])
        ])
    ])