import numpy as np
import pandas as pd

def compute_daily_return_one(data):
    """
    Return is a measure of the percentage change in price from one period to the next.
    """
    
    # Compute the return of the stock
    return_stock = np.log(data[1:]/data[:-1])
    # Add 0 at beggining to mantain the same length
    return_stock = np.insert(return_stock, 0, 0)
    
    return return_stock

def compute_daily_return(full_data, dates, companies):
    """
    Compute the daily return of a stock
    """
    
    # Get the data of the stock
    companies_return = []
    for company in companies:
        data = full_data[company].values
        return_stock = compute_daily_return_one(data)
        companies_return.append(return_stock)

    # Convert to pandas dataframe with proper indexing and date
    companies_return = pd.DataFrame(companies_return).T
    companies_return.index = dates
    companies_return.columns = companies

    return companies_return