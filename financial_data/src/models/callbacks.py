from dash import Input, Output
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import time

from src.models.libfit import find_closest_date, apply_filter_by_dates, fit_line, fit_adaptative_line

def register_callbacks(app, log_returns_difference, full_indexes, xdata_label, ydata_label, dates):
    @app.callback(
        Output('scatter-plot', 'figure'),
        Input('threshold-slider', 'value'),
        Input('date-range-slider', 'value'),
        Input('outlier-strategy', 'value')
    )
    def update_plot(threshold, range_dates, outlier_strategy):
        print('__________________________________________')
        t0 = time.time()

        initial_date = find_closest_date(pd.to_datetime(dates[range_dates[0]]), pd.to_datetime(full_indexes))
        end_date = find_closest_date(pd.to_datetime(dates[range_dates[1]]), pd.to_datetime(full_indexes))

        filtered_data = apply_filter_by_dates(log_returns_difference, initial_date, end_date)
        print("Removing outliers with method:", outlier_strategy)
        
        X = filtered_data[xdata_label].values.reshape(-1, 1)
        y = filtered_data[ydata_label].values

        x_pred, y_pred, reg_model = fit_line(X, y, nvals=100)
        residuals = y - reg_model.predict(X)

        x_pred_no_outliers, y_pred_no_outliers, accepted_idxs = fit_adaptative_line(
            X, y, residuals, initial_date, end_date, outlier_strategy, threshold
        )

        print(f"Time elapsed in preprocessing, outliers, and fitting: {time.time() - t0}")
        
        #monitor_resources()

        # Create the figure
        fig = go.Figure()
        fig.add_scatter(x=x_pred, y=y_pred, mode="markers", line=dict(color="red"), name='Fitted line')
        fig.add_trace(go.Scatter(x=X.flatten(), y=y, mode='markers', name='Raw data', marker=dict(color='blue')))
        fig.add_trace(go.Scatter(x=x_pred_no_outliers, y=y_pred_no_outliers, mode='lines', 
                                 name='Fitted line (no outliers)', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=X[~accepted_idxs].flatten(), y=y[~accepted_idxs], mode='markers', 
                                 name='Outliers', marker=dict(color='black')))
        
        fig.update_xaxes(title_text=xdata_label, range=[X.min()-np.abs(X.min())*0.1, X.max()+np.abs(X.max())*0.1])
        fig.update_yaxes(title_text=ydata_label, range=[y.min()-np.abs(y.min())*0.1, y.max()+np.abs(y.max())*0.1])
        fig.update_layout(legend=dict(orientation="h", x=0, y=-0.2))

        #monitor_resources()
        return fig
