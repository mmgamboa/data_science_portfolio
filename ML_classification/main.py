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

from src.data.get_data import get_data

train_data, test_data = get_data()

#print(train_data.head())

## Inspect dataset
# Get the number of families in PassengerId: xxxx_yy means xxxx family and yy member
aux_data_train = np.array(train_data['PassengerId'].apply(lambda x: np.array(x.split("_"), dtype=int)))
families = np.vstack(aux_data_train).T[0]
members = np.vstack(aux_data_train).T[1]
number_of_families = len(np.unique(families))
print(f"There are {number_of_families} families in the training data")
# Same plot in pie chart setting the size of figure tighter
fig = px.pie(values=np.unique(members, return_counts=True)[1], 
			 names=np.unique(members, return_counts=True)[0], 
			 title='Family sizes distribution')
# Add title to legend
fig.update_layout(legend_title_text='Family members')
fig.show()
train_data_original = train_data.copy()
print(f"There are {train_data.isna().sum(axis=0).sum()} NaN values in the training data")
print(f"There are {test_data.isna().sum(axis=0).sum()} NaN values in the test data")

