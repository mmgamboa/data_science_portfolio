import pandas as pd
from sklearn.preprocessing import StandardScaler

def apply_scaler(dataset,
                 columns_to_rescale=['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'],
                 verbose=False):

    ## Check if the columns are present in dataset
    for col in columns_to_rescale:
        if col not in dataset.columns:
            raise ValueError(f"Column {col} not present in the dataset")
        
    columns_to_keep_unchanged = list(set(dataset.columns) - set(columns_to_rescale))
    ## Step 1: Initialize the scaler
    scaler = StandardScaler()
    ## Step 2: Fit the scaler on X_train
    scaler.fit(dataset[columns_to_rescale])
    # Step 3: Transform X_train
    sub_dataset_scaled = scaler.transform(dataset[columns_to_rescale])
    # Step 4: Create a copy of X_train_ and X_test_ to avoid modifying the original DataFrames
    dataset_scaled = dataset.copy()
    # Step 5: Replace the columns_to_rescale in the copied DataFrames with the scaled values
    dataset_scaled[columns_to_rescale] = sub_dataset_scaled

    # (Optional) Convert back to DataFrame
    dataset_scaled = pd.DataFrame(dataset_scaled, 
                                columns=dataset.columns, 
                                index=dataset.index)

    columns_to_keep_unchanged = list(set(dataset.columns) - set(columns_to_rescale))
    # Convert to numeric all the columns
    dataset_scaled[columns_to_keep_unchanged] = dataset_scaled[columns_to_keep_unchanged]

    return dataset_scaled