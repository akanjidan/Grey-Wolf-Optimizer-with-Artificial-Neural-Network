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

### Combine Input Variables
Evaluate model performance by combining different numbers of variables.
```python
pt, mds = combine_input_variables(2, ranked_features1, X1, y1, '70-30 model')
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
