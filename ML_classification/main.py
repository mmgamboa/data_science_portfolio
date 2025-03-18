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

from sklearn.model_selection import train_test_split

from flask import Flask
from dash import Dash, Input, Output  

from src.data.get_data import get_data
from src.visualization.nan_distribution import nan_data
from src.visualization.layout import create_layout
from src.visualization.show_report_raw_data import inspect_data
from src.visualization.reports import input_data_figs, preproc_data_figs, nan_fig, preproc_age_fig
from src.visualization.features_report import scatter_figs
from src.models.outliers import outliers, outliers_treatment
from src.models.scaler import apply_scaler
from src.features.encoding import encode_data
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
### Get data from ./data/raw/
train_data, test_data = get_data()

### 1. Data preprocessing
## 1.1 Handling missing values
_ = nan_data(train_data, verbose=data_debug)  # just to printout the statistics
nan_counts_plot = nan_fig(train_data, verbose=data_debug)
## 1.2 Detect & remove outliers
# Columns that do not need outliers treatment: PassengerId, HomePlanet, CryoSleep, Cabin, Destination, VIP, Name, Transported 
# Columns that do need outliers treatment: Age, RoomService, FoodCourt, ShoppingMall, Spa, VRDeck
col_outliers = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
train_data_copy=train_data.__deepcopy__()
train_data_after_outliers = outliers(train_data_copy,
                                     col_outliers, 
                                     outlier_detection_method,
                                     outlier_treatment,
                                     threshold_std=outlier_threshold_std,
                                     threshold_iqr=outlier_threshold_iqr,                                    
                                     replace_by=outlier_replace_by,
                                     verbose=data_debug)
## 1.2.1 Inspect data after outliers treatment to analyze distributions
init_data_to_plot = input_data_figs(train_data, 
                            age_nbins=inspect_age_bins,
                            verbose=data_debug)
preproc_data_to_plot = input_data_figs(train_data_after_outliers, 
                                       age_nbins=inspect_age_bins,
                                       verbose=data_debug)
## 1.3 Normalize/scale numerical features if needed (StandardScaler, MinMaxScaler)
#print(train_data_after_outliers.columns)
#print(train_data_after_outliers.dtypes)
#print(train_data_after_outliers.describe())
#print(train_data_after_outliers.head())
train_data_outliers_scaled = apply_scaler(train_data_after_outliers)

## 2. Apply train, test splitting
# 2.1 Generate y array 
y_train = pd.DataFrame(train_data_outliers_scaled['Transported'])
# Drop Transported column from x_train  
train_data_outliers_scaled.drop(columns=['Transported'], inplace=True)
# 2.2 Split data
X_train, X_test, y_train, y_test = train_test_split(
    train_data_outliers_scaled, y_train, test_size=0.30, random_state=42)

## 3. Feature engineering
# 3.1 Encode categorical features
x_train, x_test = encode_data(X_train,#train_data_outliers_scaled, 
                                    X_test)#test_data)
look_at_the_age = preproc_age_fig(x_train, 
                                  x_test, 
                                  verbose=data_debug)

## Plot encoded data: correlation and PCA analysis
# TBM (to be migrated, it is already done)

## Create layout


# Plot features
#feature_figs = scatter_figs(x_train, y_train)
# Plot data
## Create server
server = Flask(__name__)
app = Dash(__name__, server=server)


app.layout = create_layout(raw_distros=init_data_to_plot,
                            preprocessed_distros=preproc_data_to_plot,
                            nan_distro=nan_counts_plot,
                            feature_distro=x_train,
                            age_distro=look_at_the_age,
                            verbose=data_debug)

# Callback to update scatter plot
@app.callback(
    Output('feature-scatter', 'figure'),
    [Input('x-feature', 'value'),
     Input('y-feature', 'value')]
)
def update_plot(x_feature, y_feature):
    fig = px.scatter(x_train, 
                     x=x_feature, 
                     y=y_feature, 
                     color=y_train["Transported"],
                     opacity=0.5,
                     title=f"{x_feature} vs {y_feature}")
    return fig


print(f"Updated at {time.localtime().tm_hour}hr {time.localtime().tm_min}min {time.localtime().tm_hour}sec")
# Callback to update age histogram when slider value changes

# Run server
if __name__ == '__main__':
    app.run_server(host=HOST, port=PORT, debug=DEBUG)