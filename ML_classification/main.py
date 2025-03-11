"""
# Spaceship Titanic

Welcome to the year 2912, where your data science skills are needed to solve a cosmic mystery. We've received a transmission from four lightyears away and things aren't looking good.

The Spaceship Titanic was an interstellar passenger liner launched a month ago. With almost 13,000 passengers on board, the vessel set out on its maiden voyage transporting emigrants from our solar system to three newly habitable exoplanets orbiting nearby stars.

While rounding Alpha Centauri en route to its first destination—the torrid 55 Cancri E—the unwary Spaceship Titanic collided with a spacetime anomaly hidden within a dust cloud. Sadly, it met a similar fate as its namesake from 1000 years before. Though the ship stayed intact, almost half of the passengers were transported to an alternate dimension!

To help rescue crews and retrieve the lost passengers, you are challenged to predict which passengers were transported by the anomaly using records recovered from the spaceship’s damaged computer system.

Help save them and change history!

# Author: Martín Gamboa
# Date: October 18th, 2024
# """

import numpy as np 
import pandas as pd 
import os
import time

import plotly.express as px
import plotly.graph_objects as go
import yaml

from flask import Flask
from dash import Dash, Input, Output  

from src.data.get_data import get_data
from src.visualization.layout import create_layout
from src.visualization.show_report_raw_data import inspect_data
from src.visualization.reports import input_data_figs, preproc_data_figs, nan_fig
from src.models.outliers import outliers, outliers_treatment
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Initial log
logging.info("Starting the Dash app...")

# Read config file
if not os.path.exists(f"./config.yaml"):
    logging.error("Configuration file not found.")
    raise FileNotFoundError("Configuration file not found.")
print(" ")
print("====================================")
with open(f"./config.yaml", 'r') as file:
    config = yaml.safe_load(file)

# Load parameters
sys_debug = config["sys_debug"]
data_debug = config["data_debug"]

if os.uname().nodename=='horizonte':
    #root_dir = os.environ["DS_DIR"]+config["paths"]["root"]
    root_dir = "./"
    if sys_debug: print(config)
    HOST = config["server"]["host"]
    PORT = config["server"]["port"]
    DEBUG = config["server"]["debug"]
else:
    root_dir = "./"
    HOST = "0.0.0.0"
    PORT = 8050
    DEBUG = config["server"]["debug"]

inspect_age_bins = config["inspect_data"]["age_bins"]
outlier_threshold_std = config["outliers"]["threshold_std"]
outlier_threshold_iqr = config["outliers"]["threshold_iqr"]
outlier_detection_method = config["outliers"]["method"]
outlier_treatment = config["outliers"]["treatment"]
outlier_replace_by = config["outliers"]["replace_by"]
#==============================================================================
#==============================================================================
# Get data from ./data/raw/
train_data, test_data = get_data()

# Inspect dataset
## Create Family distribution
init_data_to_plot = input_data_figs(train_data, 
                            age_nbins=inspect_age_bins,
                            verbose=data_debug)

# There are some columns that are not needed to treat with outliers
# such as: PassengerId, HomePlanet, CryoSleep, Cabin, Destination, VIP, Name, Transported
# So columns that will be considered for outlier detection:
#   Age, RoomService, FoodCourt, ShoppingMall, Spa, VRDeck
col_outliers = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
train_data_after_outliers = outliers(train_data,
                                     col_outliers, 
                                     outlier_detection_method,
                                     outlier_treatment,
                                     threshold_std=outlier_threshold_std,
                                     threshold_iqr=outlier_threshold_iqr,                                    
                                     replace_by=outlier_replace_by,
                                     verbose=data_debug)

# NaN's treatment
nan_counts_plot = nan_fig(train_data, verbose=data_debug)
# Plot data
## Create server
server = Flask(__name__)
app = Dash(__name__, server=server)



## Create layout
#preproc_data_to_plot = preproc_data_figs(train_data_after_outliers, 
#                                    age_nbins=inspect_age_bins,
#                                    verbose=data_debug)
preproc_data_to_plot = input_data_figs(train_data_after_outliers, 
                                       age_nbins=inspect_age_bins,
                                       verbose=data_debug)


app.layout = create_layout(raw_distros=init_data_to_plot,
                            preprocessed_distros=preproc_data_to_plot,
                            nan_distro=nan_counts_plot,
                            verbose=data_debug)
print(f"Updated at {time.localtime().tm_hour}hr {time.localtime().tm_min}min {time.localtime().tm_hour}sec")
# Callback to update age histogram when slider value changes

# Run server
if __name__ == '__main__':
    app.run_server(host=HOST, port=PORT, debug=DEBUG)