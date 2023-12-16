# Resultant Predictions Documentation

These jupyter notebooks focuses on combining predictions from two different models and calculating a weighted average to create resultant predictions. The script is applied to two scenarios: time predictions and company predictions.

## Dependencies
The following libraries are used in this script:
- **pandas, numpy**: Essential Python libraries for data manipulation and numerical computations.

## Functions
1. **resultant_preds(path1, path2)**:
   - Combines predictions from two different paths and calculates a weighted average based on predefined weights.
   - Returns a DataFrame containing the resultant predictions.

## Workflow
1. **Time Predictions**:
   - Load time predictions from two different models: 'T_CLIP' and 'T_ENET'.
   - Apply the `resultant_preds` function to combine and calculate resultant predictions.
   - Save the resultant predictions to an Excel file ('behaviour_simulation_test_time.xlsx').

2. **Company Predictions**:
   - Load company predictions from two different models: 'C_CLIP' and 'C_ENET'.
   - Apply the `resultant_preds` function to combine and calculate resultant predictions.
   - Save the resultant predictions to an Excel file ('behaviour_simulation_test_company.xlsx').

3. **Display Statistics**:
   - Display descriptive statistics of the resultant predictions for both time and company scenarios.

For further details and updates, refer to the script code and associated documentation in the provided Colab link.
