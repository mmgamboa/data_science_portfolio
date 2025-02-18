import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def find_closest_date(date, full_indexes):
    """
    Find the closest date to the one provided by the user
    """
    # Ensure full_indexes is a DatetimeIndex
    full_indexes = pd.to_datetime(full_indexes)    # Convert to datetime
    date = pd.to_datetime(date)
    # Compute the difference in days
    diff = np.abs((full_indexes - date).days)
    # Get the index of the minimum
    idx = np.argmin(diff)
    return full_indexes[idx].strftime('%Y-%m-%d')

def apply_filter_by_dates(data, initial_date, end_date):
    """ 
    Apply a filter to the data by the initial and end date
    
    Parameters
    ----------
    data: pd.DataFrame
        Data to filter
    initial_date: str
    end_date: str
    
    Returns
    -------
    pd.DataFrame
        Filtered data
        
    """
    filtered_data = data[
        (data.index >= initial_date) & 
        (data.index <= end_date)
    ]
    return pd.DataFrame(filtered_data)

def fit_line(x,y, nvals=100, verbose=True):
    """Returns the x_pred and y_pred of the linear regression
    and the model.
    
    Parameters
    ----------
    x: np.array
        x values
    y: np.array
        y values
    nvals: int
    
    Returns
    -------
    x_pred_: np.array
        x values predicted
    y_pred_: np.array
        y values predicted
    reg: LinearRegression
        Linear regression model fitted
    """
    reg = LinearRegression(fit_intercept=True).fit(x, y)

    # Get the score of the model
    if verbose:
        print(f"Score: {reg.score(x,y):10.9f}")
        print(f"Coef: {reg.coef_[0]:3.2} - {reg.intercept_:1.5f}")

    # Predict the data
    x_pred_ = np.linspace(x.min()- np.abs(x.min())*5, x.max()+ x.max()*5, nvals)
    y_pred_ = reg.predict(x_pred_.reshape(-1, 1))

    return x_pred_, y_pred_, reg

# Implement different outlier strategies
class OutlierRemover:
    
    def __init__(self, strategy, threshold):
        self.strategy = strategy
        self.threshold = threshold
    
    def std_strategy(self, data, border_cases=False):
        """
        Remove outliers using the standard deviation strategy
        
        Args:
            data (pd.Series): _description_
            border_cases (bool, optional): _description_. Defaults to False.

        Returns:
            data: pd.dataframe. Data without outliers
            accepted_idxs: pd.dataframe. Indexes of the accepted values
        """
        mean = data.mean()
        std = data.std()
    
        z = (data - mean) / std
    
        if not border_cases:
            accepted_idxs = abs(z) < self.threshold
        else:
            accepted_idxs = abs(z) < self.threshold + 0.1*self.threshold
        data = data[accepted_idxs]
    
        return data, accepted_idxs
    
    def iqr_strategy(self, data, border_cases=False):
        """
        Remove outliers using the IQR strategy
        
        Args:
        data: pd.Series
        border_cases: bool
        
        Returns:
        data without outliers: pd.Series
        accepted_idxs: pd.Series       
        
        If border_cases = True: consider the 10% greater and 10% lower values
        This is useful for border cases where the data is not normally distributed
        """
        q1 = pd.DataFrame(data).quantile(0.25).iloc[0]
        q3 = pd.DataFrame(data).quantile(0.75).iloc[0]
        
        iqr = q3 - q1        
        
        lower_bound = q1 - self.threshold * iqr
        upper_bound = q3 + self.threshold * iqr
        
        if not border_cases:
            accepted_idxs = (data > lower_bound) & (data < upper_bound) 
        else:
            accepted_idxs = (data > lower_bound - 0.1*lower_bound) & (data < upper_bound + 0.1*upper_bound)
            
        data = data[accepted_idxs]
    
        return data, accepted_idxs
    
    def remove_outliers(self, data, border_cases=False):
        """
        
        
        """
        if self.strategy == 'std':
            return self.std_strategy(data, border_cases=border_cases)
        elif self.strategy == 'iqr':
            return self.iqr_strategy(data, border_cases=border_cases)
    
def fit_adaptative_line(X, y, residuals, initial_date, end_date, outlier_strategy, threshold):
    """
    Fit a line to the data considering the outliers
    
    Parameters
    ----------
    X: np.array
        x values
    y: np.array
        y values
    residuals: np.array
        residuals of the model
    initial_date: str
    end_date: str
    outlier_strategy: str
        'std' or 'iqr'
    threshold: float
    
    Returns
    -------
    x_pred_no_outliers: np.array
        x values predicted without outliers
    y_pred_no_outliers: np.array
        y values predicted without outliers
    accepted_idxs: np.array
        indexes of the accepted values   
    """
    
    
    outlier_removal = OutlierRemover(outlier_strategy, threshold)

    _, accepted_idxs = outlier_removal.remove_outliers(residuals, border_cases=False)
    X_no_outliers = X[accepted_idxs]
    y_no_outliers = y[accepted_idxs]

    print(f"Fitting line from {initial_date} to {end_date}. ======================")
    
    # Fit the model without outliers
    x_pred_no_outliers, y_pred_no_outliers, fitted_model = fit_line(X_no_outliers, y_no_outliers, nvals=100)
    score_fix_border = fitted_model.score(X_no_outliers, y_no_outliers)
    
    ### Compute Score for linear regression considering 10% greater the outlier limit to account for border cases if improve the score
    _, accepted_idxs_var = outlier_removal.remove_outliers(residuals, border_cases=True)
    X_no_outliers_var = X[accepted_idxs_var]
    y_no_outliers_var = y[accepted_idxs_var]
    # Fit the model without outliers
    x_pred_no_outliers_var, y_pred_no_outliers_var, fitted_model_var = fit_line(X_no_outliers_var,
                                                                                y_no_outliers_var, 
                                                                                nvals=100)
    score_var_border = fitted_model_var.score(X_no_outliers_var, y_no_outliers_var)
    if score_fix_border >= score_var_border:
        x_pred_no_outliers = x_pred_no_outliers
        y_pred_no_outliers = y_pred_no_outliers
        accepted_idxs = accepted_idxs
    else:
        x_pred_no_outliers = x_pred_no_outliers_var
        y_pred_no_outliers = y_pred_no_outliers_var
        accepted_idxs = accepted_idxs_var
        print("!Outliers account for Fitting computation!")   
        
    return  x_pred_no_outliers, y_pred_no_outliers, accepted_idxs

