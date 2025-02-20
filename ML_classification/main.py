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

import plotly.express as px
import plotly.graph_objects as go
import yaml

from flask import Flask
from dash import Dash, Input, Output  

from src.data.get_data import get_data
from src.visualization.layout import create_layout
from src.visualization.show_report_raw_data import inspect_data
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

sys_debug = config["sys_debug"]

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

# Get data from ./data/raw/
train_data, test_data = get_data()


# Inspect dataset
## Create Family distribution
(family_distro, 
 cryo_sleep_distro,
 homeplanet_distro,
 destinatioplanet_distro,
 age_distro) = inspect_data(train_data, 
                            age_nbins=50,
                             verbose=False)

## Create server
server = Flask(__name__)
app = Dash(__name__, server=server)

app.layout = create_layout(family_distro, 
                           cryo_sleep_distro,
                           homeplanet_distro,
                           destinatioplanet_distro,
                           age_distro)
# Callback to update age histogram when slider value changes
@app.callback(
    Output('age-histogram', 'figure'),
    Input('nbins-slider', 'value')
)
def update_age_histogram(nbins):
    fig = px.histogram(train_data, x="Age", nbins=nbins, title="Age Distribution")
    # Display the number of bins within fig plot
        
    return fig

# Run server
if __name__ == '__main__':
    app.run_server(host=HOST, port=PORT, debug=DEBUG)