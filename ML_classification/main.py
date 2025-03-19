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
from src.visualization.features_report import feature_selection_figs
from src.models.outliers import outliers, outliers_treatment
from src.models.scaler import apply_scaler
from src.features.encoding import encode_data
from src.visualization.callbacks import register_callbacks

# ML models
from src.models.classification import ClassificationModel
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

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
_ = nan_data(train_data.copy(), 
             verbose=data_debug)  # just to printout the statistics
nan_counts_plot = nan_fig(train_data.copy(), 
                          verbose=data_debug)
## 1.2 Detect & remove outliers
# Columns that do not need outliers treatment: PassengerId, HomePlanet, CryoSleep, Cabin, Destination, VIP, Name, Transported 
# Columns that do need outliers treatment: Age, RoomService, FoodCourt, ShoppingMall, Spa, VRDeck
col_outliers = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
#train_data_copy=train_data.__deepcopy__()
train_data_after_outliers = outliers(train_data.copy(),
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

## 4. Feature selection: Correlation and PCA analysis
corr_pca_figs = feature_selection_figs(x_train, y_train)
# Drop non-relevant components

## 5. Model selection
# 5.1 Train a model
xgmodel = ClassificationModel(which_model='xgboost')
model_trained = xgmodel.train(x_train, y_train)
# 5.2 Evaluate the model and # 5.3 Asses performance
print(model_trained)
predictions = model_trained.predict(x_test.drop(columns=['PassengerId', 'Name']))
def compute_metrics(y_true, y_pred, mode='binary', y_prob=None):
    full_metrics={}
    full_metrics['precision'] = precision_score(y_true, y_pred, average=mode)
    full_metrics['recall'] = recall_score(y_true, y_pred, average=mode)
    full_metrics['f1_score'] = f1_score(y_true, y_pred, average=mode)
    full_metrics['accuracy'] = accuracy_score(y_true, y_pred)
    if y_prob is None:
        full_metrics['roc_auc'] = roc_auc_score(y_true, y_pred)
    else: 
        full_metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
    # Confusion matrix
    full_metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    # Classification report
    full_metrics['classification_report'] = classification_report(y_true, y_pred)
    
    def plot_confusion_matrix(metrics):
        conf_matrix = metrics['confusion_matrix']
        conf_matrix_fig = px.imshow(conf_matrix, 
                                    labels=dict(x="Predicted", y="Actual", color="Count"),
                                    x=['Not Transported', 'Transported'],
                                    y=['Not Transported', 'Transported'],
                                    title="Confusion Matrix")

        conf_matrix_fig.update_layout(coloraxis_showscale=False)
        
        return conf_matrix_fig
    
    full_metrics['confusion_matrix_fig'] = plot_confusion_matrix(full_metrics)
    
    return full_metrics

metrics = compute_metrics(y_test, predictions)
# Plot Confusion Matrix



## Create layout
# Create server
server = Flask(__name__)
app = Dash(__name__, server=server)
# Callbacks for interactive plots
register_callbacks(app, train_data, x_train, y_train)
app.layout = create_layout(raw_distros=init_data_to_plot,
                            preprocessed_distros=preproc_data_to_plot,
                            nan_distro=nan_counts_plot,
                            feature_distro=x_train,
                            age_distro=look_at_the_age,
                            corr_pca_figs=corr_pca_figs,
                            raw_data=train_data,
                            ml_model=metrics,
                            verbose=data_debug)

print(f"Updated at {time.localtime().tm_hour}hr {time.localtime().tm_min}min {time.localtime().tm_hour}sec")

# Run server
if __name__ == '__main__':
    app.run_server(host=HOST, port=PORT, debug=DEBUG)