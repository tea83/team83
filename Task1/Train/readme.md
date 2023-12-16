# XGBoost Training and Inference with K-Fold Cross-Validation

This Python script demonstrates the training and inference process using XGBoost with K-Fold Cross-Validation. The code includes GPU memory management, model training for each fold, and out-of-fold (OOF) prediction evaluation.

## Functions and Classes

### `IterLoadForDMatrix` Class

A custom iterator for loading data into a `DeviceQuantileDMatrix` in XGBoost.

#### Constructor:

- `df` (pd.DataFrame): DataFrame containing the data.
- `features` (list): List of feature column names.
- `target` (str): Name of the target column.
- `batch_size` (int): Size of each batch (default is 256 * 1024).

#### Methods:

- `reset()`: Reset the iterator.
- `next(input_data)`: Yield the next batch of data.

### Main Training Loop

- Convert the cudf DataFrame to a pandas DataFrame.
- Define a subsample ratio for training (`TRAIN_SUBSAMPLE`).
- Perform garbage collection to free up GPU memory.
- Initialize KFold with the specified number of folds (`FOLDS`).
- Loop through each fold in the KFold cross-validation.
    - Train with a subsample of the training fold data if specified.
    - Create training, validation, and test datasets for the current fold.
    - Train the XGBoost model for the current fold.
    - Save the trained model for the current fold.
    - Infer and evaluate out-of-fold (OOF) predictions for the current fold.
    - Save the OOF predictions.
    - Clean up variables to free memory.

