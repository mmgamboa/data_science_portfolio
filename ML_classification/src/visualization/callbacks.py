# callbacks.py
import plotly.express as px
from dash import Input, Output

def register_callbacks(app, train_data, x_train, y_train):
    @app.callback(
        Output('input-feature-scatter', 'figure'),
        [Input('input-x-feature', 'value'),
         Input('input-y-feature', 'value'),
         Input("x-scale-toggle", "value"),
         Input("y-scale-toggle", "value")]
    )
    def update_plot(raw_x_feature, raw_y_feature, x_scale_type, y_scale_type):
        fig = px.scatter(train_data, 
                         x=raw_x_feature, 
                         y=raw_y_feature, 
                         color=train_data["Transported"],
                         opacity=0.5,
                         title=f"{raw_x_feature} vs {raw_y_feature}")
        fig.update_layout(yaxis_type=y_scale_type, xaxis_type=x_scale_type)
        return fig

    @app.callback(
        Output('feature-scatter', 'figure'),
        [Input('x-feature', 'value'),
         Input('y-feature', 'value')]
    )
    def update_plot(x_feature, y_feature):
        fig = px.scatter(x_train.copy(), 
                         x=x_feature, 
                         y=y_feature, 
                         color=y_train["Transported"],
                         opacity=0.5,
                         title=f"{x_feature} vs {y_feature}")
        return fig