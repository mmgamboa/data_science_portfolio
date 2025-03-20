import numpy as np 
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

def feature_selection_figs(X, y, verbose=False):
    
    X_copy = X.copy()
    X_copy.drop(columns=['PassengerId', 'Age', 
                         'Name', #'TotalServices'
                         ], 
                inplace=True)

    correlation_fig = correlation_matrix_fig(X_copy, y, verbose=verbose)
    pca_fig = pca_analysis_fig(X_copy, y, verbose=verbose)
    return correlation_fig, pca_fig
    
def correlation_matrix_fig(X, y,
                           threshold=0.85,
                           verbose=False):

    data_to_inspect=X#.loc[y[y==True].index]
    
    # Compute the correlation matrix with the transported passengers
    corr_matrix = data_to_inspect.corr()

    # Plotting the correlation matrix as a heatmap
    # Convert correlation matrix to Plotly heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,  # Correlation values
        x=corr_matrix.columns,  # Feature names
        y=corr_matrix.index,  # Feature names
        colorscale="rdbu_r",  # Color scheme
        zmin=-1, zmax=1,  # Ensures proper scaling
        colorbar=dict(title="Correlation"),
        hoverongaps=False
    ))

    # Update layout
    fig.update_layout(
        title="Correlation Matrix Heatmap",
        xaxis=dict(title="Features"),
        yaxis=dict(title="Features", autorange="reversed"),  # Ensures correct order
        template="plotly_white"
    )

    # Find pairs of features that are highly correlated
    high_corr_pairs = np.where(np.abs(corr_matrix) > threshold)
    high_corr_pairs = [(corr_matrix.index[x], corr_matrix.columns[y]) for x, y in zip(*high_corr_pairs) if x != y and x < y]

    if verbose:
        print("Highly correlated feature pairs (above the threshold):")
        for pair in high_corr_pairs:
            print(pair)

    return fig

def pca_analysis_fig(X, y, verbose=False):
    ##########################
    ##########
    ########## Applying Preprocessing to the data.
    ##########
    ###########################
    
    #data_to_inspect=X.loc[y[y==True].index]
    # Step 1: Standardize the data
    scaler = StandardScaler()
    X_scaled_ = scaler.fit_transform(X)
    X_scaled = np.nan_to_num(X_scaled_, nan=-1.0)
    # Step 2: Apply PCA
    pca = PCA()  # You can also specify the number of components like PCA(n_components=5)
    X_new = pca.fit_transform(X_scaled)#.dropna())

    # Step 3: Analyze the explained variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    if verbose: 
        print(f"Explained variance: {explained_variance}")
    # Step 4: Create an interactive Plotly figure
    fig = go.Figure()

    # Add bar chart for individual explained variance
    fig.add_trace(go.Bar(
        x=[f"PC{i+1}" for i in range(len(explained_variance))],
        y=explained_variance, 
        name="Individual Explained Variance",
        marker=dict(color="blue", opacity=0.6)
    ))

    # Add step line for cumulative explained variance
    fig.add_trace(go.Scatter(
        x=[f"PC{i+1}" for i in range(len(cumulative_variance))],
        y=cumulative_variance, 
        mode="lines+markers",
        name="Cumulative Explained Variance",
        line=dict(color="red", width=2, dash="dash")
    ))

    # Update layout
    fig.update_layout(
        title="Explained Variance by Principal Components",
        xaxis_title="Principal Components",
        yaxis_title="Explained Variance Ratio",
        xaxis=dict(tickmode="linear"),
        yaxis=dict(range=[0, 1.1]),  # Ensures variance is properly scaled
        legend=dict(x=0.7, y=0.3),
        template="plotly_white"
    )
    # Step 5: Select the number of components (e.g., for pctg% variance)
    pctg=0.97
    n_components = np.argmax(cumulative_variance >= pctg) + 1
    if verbose: 
        print(f'Number of components to retain {pctg*100}% variance: {n_components} out of {X_scaled.shape[1]}')

    return fig