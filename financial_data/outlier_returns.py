"""
Author: Martin Gamboa
GitHub: mmgamboa
Date: January 10th, 2025

# Problem 1: Handling Outliers in Regression or Chi-Square Fitting

**Objective**. Explore different alternatives to determine whether an outlier should be considered or discarded when performing linear regression or chi-square fitting. Special attention is required for borderline cases where it is not evident if the point should be excluded.

**Requirements**

* Avoid using smoothness techniques.
* Effectiveness is not the priority; computational resource requirements for daily computation must be explicitly stated.

**Data**.
* QQQ and IWM (used as benchmarks).
* 2YM (likely referring to a 2-year metric or dataset).
* Use a logarithmic scale for computations.


"""
# Load specific packages
import os
import yaml

import plotly.express as px

from dash import Dash

# Import local module
from src.data.get_data import load_data
from src.features.build_features import compute_daily_return
from src.visualization.plot_lib import (plot_scatter_returns, 
                                        plot_log_return_difference)
from src.models.layout import create_layout
from src.models.callbacks import register_callbacks

import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Example usage
logging.info("Starting the Dash app...")

# Load configutration file
with open(f"./financial_data/config.yaml", 'r') as file:
    config = yaml.safe_load(file)

# Extract the parameters needed for running script from the configuration file
companies_name = config['companies_fit']
period = config["download_params"]["period"]
interval = config["download_params"]["interval"]
start_date = config["download_params"]["start_date"]
end_date = config["download_params"]["end_date"]
param_to_analyze = config["financial_param"]
# Paths
root_dir = "./"+config["paths"]["root"]
raw_data_dir = config["paths"]["raw"]
PATH_RAW_DIR = root_dir+"/"+raw_data_dir
# Plots
savefigs = config["savefigs"]
reports_dir = config["paths"]["reports"]
PATH_REPORTS_DIR = root_dir+"/"+reports_dir
plot_verbosity = config["plot_verbosity"]
# Server setup
HOST = config["server"]["host"]
PORT = config["server"]["port"]
DEBUG = config["server"]["debug"]
# Import the download method from get_data.py

#############################################################
######
######      The following code is the main code
######
#############################################################
# Use the imported method to download the data
data = load_data(companies_name, 
                period, 
                interval,
                PATH_RAW_DIR,
                start=start_date,
                end=end_date)

# Normalize data to start at 1
data = data[param_to_analyze]
data = data / data.iloc[0]

compname1 = companies_name[0]
compname2 = companies_name[1]

# Plot trends
if plot_verbosity:
    fig = px.line(data, title=f"Normalized price of {compname1} and {compname2}")
    # Show and save plot
    if savefigs:
        fig.write_image(f"{PATH_REPORTS_DIR}/normalize_price_{compname1}_{compname2}.jpeg")    
    fig.show()

# Look at the daily return of elected companies
log_returns_difference = compute_daily_return(data, 
                                              data.index, 
                                              [compname1, compname2])

if plot_verbosity:
    plot_log_return_difference(log_returns_difference, 
                               [compname1, compname2],
                               PATH_REPORTS_DIR=PATH_REPORTS_DIR,
                               savefig=savefigs)
    plot_scatter_returns(log_returns_difference, 
                         [compname1, compname2],
                         PATH_REPORTS_DIR=PATH_REPORTS_DIR,
                         savefig=savefigs)

# Initialize Dash app
app = Dash(__name__)

# Load data (ensure you define these variables)
dates = data.index.values  # Load your dates
date_indices = {i: date for i, date in enumerate(dates)}
full_indexes = log_returns_difference.index.values
xdata_label = compname1
ydata_label = compname2

# Set app layout
app.layout = create_layout(dates, 
                           date_indices)

# Register callbacks
register_callbacks(app, 
                   log_returns_difference, 
                   full_indexes, 
                   xdata_label, 
                   ydata_label, 
                   dates)

# Run server
if __name__ == '__main__':
    app.run_server(host=HOST, port=PORT, debug=DEBUG)
