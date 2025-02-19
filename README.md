# Data Science Portfolio

**Author**: [Martín Gamboa](https://www.linkedin.com/in/martin-gamboa/).

## Table of contents:
- [Introduction](#Introduction)
- [Handling outliers](#Outliers)

## Introduction
This repository contains a collection of tools, scripts, and Jupyter Notebooks designed to showcase my technical knowledge, tools, and programming style. The repository is primarily for recruiting, HR, and technical teams at companies or organizations that are looking for data scientists, data analysts, data engineers, technical support, or researchers, among other possible positions.

You will find stages of data pre-processing, visualization, engineering, generative AI models, ML, DL, tuning, classifiers, etc.

## Outliers 

### Live Demo 🎉  
Check out the live app of `financial_data` here. I am using Render free instance. So it will spin down with inactivity, which can delay requests by 50 seconds or more.

**Situation**. Generate a confident model to fit the trend between the stock prices of two assets. 

**Task**. Create a linear regression model to fit a line. Determine whether an outlier should be considered or discarded when performing linear regression. Special attention is required for borderline cases where it is unclear if the point should be excluded.

**Action**. Consider two methods to determine outliers: `std` and `IQR`. We will use typical values to discard outliers and apply the model. Consider the accuracy and re-compute the model and the metric accounting border cases within a user-determined threshold. **Generate an easy-to-use** app and deploy it in the cloud. 

**Results**. [![View Dashboard](https://img.shields.io/badge/Live_App-Click_Here-brightgreen)](https://financial-data-vybw.onrender.com/)

**Data**.
* QQQ and IWM (used as benchmarks).
* 2YM (likely referring to a 2-year metric or dataset).
* Use a logarithmic scale for computations.
