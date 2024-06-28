# Grey-Wolf-Optimizer-with-Artificial-Neural-Network

# GWO_ANN_final_version

This project uses Grey Wolf Optimization (GWO) to optimize feature selection for an Artificial Neural Network (ANN) regressor. The workflow includes data preprocessing, analysis, and the application of GWO for feature selection, followed by training and evaluating the ANN model.

## Installation and Setup

### Prerequisites
- Python 3.6 or later
- Jupyter Notebook
- Required libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `imblearn`, `itertools`, `gdown`

### Installation
1. Clone the repository or download the `.ipynb` file.
2. Install required packages using pip:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn gdown
    ```
3. Download the dataset using the provided link in the notebook.

### Dataset
The dataset used is an Excel file (`All_data2.xlsx`). Ensure it is available in the same directory as the notebook or update the file path accordingly.

## Data Analysis and Preprocessing

### Step 1: Load the Data
```python
import pandas as pd

file_path = '/content/All_data2.xlsx'
data = pd.read_excel(file_path)
print("Data Columns:", data.columns)
data.head()
```

### Step 2: Handle Missing Values
- Identify numerical and categorical columns.
- Impute missing values using mean for numerical columns and the most frequent value for categorical columns.

### Step 3: Data Cleaning and Analysis
- Convert relevant columns to numeric.
- Display basic information and summary statistics.
- Visualize distributions to identify outliers.

## Functions

### Visualization Functions
- `make_r_plot(y_test_tensor, y_pred)`: Creates a regression plot with R² value.
- `pltloss(mlp)`: Plots the training loss curve.
- `plotbar(ranked_features, feature_names)`: Displays a bar chart of feature importance.

### Model Training and Evaluation
- `combine_input_variables(combination_num, ranked_features, X, y, model_name)`: Evaluates model performance with different combinations of input variables.

## Grey Wolf Optimization (GWO)

### GWO Class
Implements the GWO algorithm for feature selection.
- **Initialization**: Sets up the population of wolves, their positions, and scores.
- **Fitness Function**: Uses ANN to evaluate the fitness of feature subsets.
- **Update Position**: Updates wolf positions based on alpha, beta, and delta wolves.
- **Binarize**: Converts continuous position values to binary to determine selected features.
- **Optimization Loop**: Iteratively updates positions and evaluates fitness.

### Usage
1. Initialize the GWO class with the number of features, wolves, and iterations.
2. Optimize to select the best features.
3. Evaluate the importance of each selected feature.
4. Rank the features by importance.

## Example Usage

### Data Split and Training
The data split are in 70-30 split which trains on the traditional data. The other split is teh 80-2- split which trains on data that has its feature been engineered. 
For the 70-30 split, it follows this procedure

```python
X = data_all.iloc[:, :-1].values
y = data_all.iloc[:, -1].values.astype(float)
num_features = X.shape[1]

gwo = GWO(num_features, num_wolves=10, max_iter=30)
selected_features = gwo.optimize(X, y, test_size=0.3)
print("Selected features:", selected_features)

# Evaluate and rank the importance of each selected feature
feature_importance = {feature: mean_squared_error(y_test, mlp.predict(X_test)) for feature in selected_features}
ranked_features = sorted(feature_importance, key=feature_importance.get)
plotbar(ranked_features, feature_names)
```
For the 80-20 split data, we have:
1. **Data Loading and Preprocessing**
   - The dataset is loaded from an Excel file.
   - `SimpleImputer` is used to handle missing values by replacing them with the mean.
   - `StandardScaler` is used to normalize the features, although it is commented out in some parts of the code.

2. **80-20 Split**
   - The dataset is split into training and testing sets using an 80-20 ratio.
   - Features are selected using the `GWO` (Grey Wolf Optimizer) algorithm.
   - The importance of each selected feature is evaluated and ranked based on Mean Squared Error (MSE).

3. **Model Training and Evaluation**
   - An MLPRegressor is trained on the selected features.
   - Various performance metrics such as MAE, MSE, RMSE, SI, and R-Value are calculated and displayed.
   - The performance metrics and selected features are appended to a results DataFrame.
   - Loss and R plots are generated using the `pltloss` and `make_r_plot` functions.

4. **Feature Engineering**
   - New interaction features are created and added to the dataset.
     - `Area_Floor_Interaction`
     - `Log_Total_Area`
     - `Sqrt_Number_of_Floor`
     - `Duration_Type_Interaction`
     - `Location_Access_Interaction`
   - These new features are added to the dataset, and the feature selection and model evaluation process is repeated.
   - Results and performance metrics for the engineered features are appended to the results DataFrame.


### Combine Input Variables
**Combining Features**
Evaluate model performance by combining different numbers of variables.
   - The `combine_input_variables` function is used to combine two, three, and four input variables.
   - The results for these combined inputs are appended to the results DataFrame.

```python
pt, mds = combine_input_variables(num, ranked_features1, X1, y1, '70-30 model')
# num >= max(num of input columns) is the number of variables you want to combine
all_results = pd.concat([all_results, mds], ignore_index=True)
```

## Results
The results are stored in a DataFrame and include:
- Model name
- Selected features
- Test loss
- MAE, MSE, RMSE, SI, R-Value

## Conclusion
This notebook demonstrates the effective use of GWO for feature selection in an ANN regressor model, showcasing the combination of optimization algorithms with machine learning for improved model performance.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
