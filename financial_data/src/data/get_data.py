import yfinance as yf
import pandas as pd

def load_data(companies, 
              period, 
              interval,
              PATH_DIR,
              start="2024-01-01",
              end="2025-02-01"):
    """
    Load data from Yahoo Finance API. If the data is not in the cache, it will be downloaded and saved in the cache.
    
    Parameters:
    companies: list of str
        List of companies to download data
    period: str
        Period to download data
    interval: str
        Interval to download data
    start: str
        Start date
    end: str
        End date
    Returns:
    data: pd.DataFrame
        DataFrame with the data    
    """
    print("Downloading data of the following companies: ", companies)
    print("Period: ", period)
    print("Interval: ", interval)
    print("Ranging from ", start, " to ", end)
    
    # Sort and join companies names
    companies_name = "_".join(sorted(companies))
    # Remove '-' in start and end dates
    start_name = str(start).replace("-", "")
    end_name = str(end).replace("-", "")
    # Make name in format COMPANIES_START_END
    name = companies_name + "_" + start_name + "_" + end_name
    
    # Path to data
    path = lambda name: PATH_DIR+f"{name}.csv"
    # List files in cache
    
    try:
        data = pd.read_csv(path(name), header=[0, 1], index_col=0)
        print("Data loaded from cache.")
        # Set the first column as the index
        return data
    except:
        print("Data not found in cache.")
        data = yf.download(companies, 
                           period=period, 
                           interval=interval,
                           start=start,
                           end=end)
        # Save data in cache
        data.to_csv(path(name), index=True)
        return data