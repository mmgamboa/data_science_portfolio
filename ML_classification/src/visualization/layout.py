from dash import html, dcc
import pandas as pd


def create_layout(family_distro,
                  cryo_sleep_distro,
                  homeplanet_distro,
                  destinationplanet_distro,
                  age_distro,
                  verbose =False):
    if verbose: 
        print("Received figure type:", type(family_distro))
    return html.Div([
        html.H1("Inspecting Passengers data"),
        # Tabs for different plots
        dcc.Tabs([
            dcc.Tab(label="Inspecting Raw Data", children=[
                dcc.Graph(figure=family_distro),
                dcc.Graph(figure=cryo_sleep_distro),
                dcc.Graph(figure=homeplanet_distro),
                dcc.Graph(figure=destinationplanet_distro),

                # Histogram with dynamic bin selection
                #html.Label("Select Number of Bins for Age Histogram:"),
                #dcc.Slider(id='nbins-slider', min=2, max=50, step=1, value=10,
                #           marks={i: str(i) for i in range(2, 51, 10)}),
                dcc.Graph(id='age-histogram', figure=age_distro)
            ]),
            dcc.Tab(label="Inspecting Cleaned Data", children=[
                html.H1("Inspecting Cleaned Data")
            ])
        ])
    ])