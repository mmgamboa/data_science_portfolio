"""
Author: Martin Gamboa
GitHub: mmgamboa
Date: January 10th, 2025


# Problem

What does high correlation of returns imply for long-term behavior of two assets prices? Analyze some examples to support your answer. You can get price data from yahoo finance.

# Solution
_Get data_

* Get the data from external source using Yahoo Finance,

_Trends_

* Define Return computation for individual data and dataset. We will use log return for continuous compounding and 'Adj Close' column. This election is because we account for stock, dividends, and other corporate actions, providing a more accurate representation of the stock's performance over time,

* We plot using `plotly` and with dynamic box showing simultaneously the return for all the companies considered for this analysis. I did it twice: one step-by-step developing my own functions and another using built-in methods from `Pandas` (see _Same using built-in methods_ sub-section).
"""

# Load mymodule
import sys
import yaml

# Import local module
from src.data.get_data import load_data
from src.features.build_features import compute_daily_return
from src.visualization.plot_lib import (trends_from_dataframe, 
                                        correlation_heatmap)

sys.path.append('..')

# Load the configuration file
with open('/home/mgamboalerena/Documentos/data_science_portfolio/financial_data/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Extract the parameters needed for running script from the configuration file
companies_name = config['companies_name']
period = config["download_params"]["period"]
interval = config["download_params"]["interval"]
start_date = config["download_params"]["start_date"]
end_date = config["download_params"]["end_date"]
param_to_analyze = config["financial_param"]
root_dir = config["paths"]["root"]
raw_data_dir = config["paths"]["raw"]
PATH_RAW_DIR = root_dir+"/"+raw_data_dir
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

# Trends from raw data
returns_of_companies = compute_daily_return(data[param_to_analyze], 
                                            data.index, companies_name)

# Plot the data
trends_from_dataframe(returns_of_companies, title = 'Stock Return')

# Plot correlation matrix
correlation_heatmap(returns_of_companies)
# 
