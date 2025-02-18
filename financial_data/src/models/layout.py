from dash import html, dcc
import pandas as pd

#def create_layout(dates, date_indices):
#    return html.Div([
#        html.H1("Interactive Scatter Plot with Fitted Line"),
#        
#        dcc.Graph(id='scatter-plot'),
#        
#        html.Label("Threshold for Outlier Detection:"),
#        dcc.Slider(id='threshold-slider', 
#                   min=1, max=5, step=0.5, value=1.5,
#                   marks={i: str(i) for i in range(1, 6)}),
#        
#        html.Label("Select Initial Date:"),
#        dcc.RangeSlider(id='date-range-slider', 
#                        min=0, max=len(dates)-1, step=1, 
#                        value=[0, len(dates)-1],
#                        marks={i: date_indices[i] for i in range(0, len(dates), 30)}),
#        
#        html.Div([
#            dcc.Dropdown(['std', 'iqr'], id='outlier-strategy', value='std'),
#        ]),
#    ])
def create_layout(dates, date_indices, fig_log_return, fig_scatter_return):
    return html.Div([
        html.H1("Financial Data Analysis"),
        
        # Tabs for different plots
        dcc.Tabs([
            dcc.Tab(label="Interactive Scatter Plot", children=[
                dcc.Graph(id='scatter-plot'),
                
                html.Label("Threshold for Outlier Detection:"),
                dcc.Slider(id='threshold-slider', 
                           min=1, max=5, step=0.5, value=1.5,
                           marks={i: str(i) for i in range(1, 6)}),
                
                html.Label("Select Initial Date:"),
                dcc.RangeSlider(id='date-range-slider', 
                                min=0, max=len(dates)-1, step=1, 
                                value=[0, len(dates)-1],
                                marks={i: date_indices[i] for i in range(0, len(dates), 30)}),
                
                html.Div([
                    dcc.Dropdown(['std', 'iqr'], id='outlier-strategy', value='std'),
                ]),
            ]),
            dcc.Tab(label="Log Return Difference", children=[
                dcc.Graph(figure=fig_log_return)  # First Tab
            ]),
            dcc.Tab(label="Scatter Returns", children=[
                dcc.Graph(figure=fig_scatter_return)  # Second Tab
            ])
        ])
    ])