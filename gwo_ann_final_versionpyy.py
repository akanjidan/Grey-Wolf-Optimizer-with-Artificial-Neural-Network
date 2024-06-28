# -*- coding: utf-8 -*-


!gdown 1Zi88cQmN_6Wn-tBTQS7N6qUzzj1KU1jY

"""# DATA ANALYSIS"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
# Step 1: Load the Data and Understand the Columns

# Load the data from the Excel file
file_path = '/content/All_data2.xlsx'
data = pd.read_excel(file_path)

# Display the first few rows and column names to understand the data
print("Data Columns:", data.columns)
data.head()

from sklearn.impute import SimpleImputer

data2=data
# Check which columns are numerical and which are categorical
numerical_cols = data2.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_cols = data2.select_dtypes(include=['object']).columns.tolist()

print(f"Numerical columns: {numerical_cols}")
print(f"Categorical columns: {categorical_cols}")

# Impute missing values
if numerical_cols:
    # Impute numerical columns with mean
    imputer_num = SimpleImputer(strategy='mean')
    data2[numerical_cols] = imputer_num.fit_transform(data2[numerical_cols])

if categorical_cols:
    # Impute categorical columns with the most frequent value
    imputer_cat = SimpleImputer(strategy='most_frequent')
    data2[categorical_cols] = imputer_cat.fit_transform(data2[categorical_cols])

# Verify that there are no more missing values
print("Missing values after imputation:")
# print(X.isnull().sum())
data_all = data2.drop(index=0).reset_index(drop=True)
# data_all=X
# X=[]
data_all= data_all.dropna()
data_all.head()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Data Analysis and Problem Identification

data_cleaned = data_all

# Convert relevant columns to numeric, coercing errors to NaN
numeric_columns = data.columns.values.tolist()

for col in numeric_columns:
    data_cleaned[col] = pd.to_numeric(data_cleaned[col], errors='coerce')

# Display basic information about the cleaned dataset
data_cleaned.info()

# Check for missing values
missing_values = data_cleaned.isnull().sum()
print("\nMissing Values:\n", missing_values)

# Display summary statistics for numerical columns
summary_stats = data_cleaned.describe()
print("\nSummary Statistics:\n", summary_stats)

# # Check for unique values and distributions in categorical columns
# categorical_columns = ['Construction (1) / Demolition (2)', 'Year of construct', 'location',
#                        'Type of building', 'Number of floor', 'Site access']
# for col in categorical_columns:
#     print(f"\nUnique values in {col}:\n", data_cleaned[col].value_counts())

# Plot distributions of numerical columns to identify outliers
for col in numeric_columns:
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=data_cleaned[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

"""# RELEVANT FUNCTION"""

def make_r_plot(y_test_tensor,y_pred):
  plt.figure(figsize=(6, 6))
  plt.scatter(y_test_tensor, y_pred, c='crimson')
  # try:
  coeff = np.polyfit(y_test_tensor, y_pred, 1)
  slope, intercept = coeff
  reg_line = slope * y_test_tensor + intercept
  plt.plot(y_test_tensor, reg_line, 'b-')
  # Add the equation of the regression line
  # print(slope,intercept)
  equation = f'Regression Line: y = {slope:.2f}x + {intercept:.2f}'
  plt.text(0.05, 0.85, equation, fontsize=8, transform=plt.gca().transAxes)
  # Calculate R² using r2_score on the actual and predicted values
  r_squared = r2_score(y_test_tensor, y_pred)
  # Place the R² text after calculating it
  plt.text(0.05, 0.9, 'R²: {:.2f}'.format(r_squared), fontsize=9,
  transform=plt.gca().transAxes)
  # except Exception as e:
  #   print(f"Error in regression line fitting: {e}")

  plt.xlabel('Actual EAC', fontsize=9)
  plt.ylabel('Predicted EAC', fontsize=9)
  plt.axis('equal')
  plt.show()

# Plotting the training loss curve
def pltloss(mlp):
  plt.figure(figsize=(10, 6))
  plt.plot(mlp.loss_curve_)
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.title('Training Loss Curve')
  plt.show()


def plotbar(ranked_features, feature_names):
  plt.figure(figsize=(10, 6))
  plt.bar(range(len(ranked_features)), [feature_importance[feature] for feature in ranked_features])
  plt.xticks(range(len(ranked_features)), [feature_names[i] for i in ranked_features], rotation=90)
  plt.xlabel('Features')
  plt.ylabel('MSE')
  plt.title('Feature Importance (MSE)')
  plt.show()


def combine_input_variables(combination_num, ranked_features,X,y,model_name):
  if int(combination_num) <= int(len(ranked_features)):
    combinations = list(itertools.combinations(ranked_features , combination_num))
    # Initialize results DataFrame
    results = pd.DataFrame(columns=['Features', 'Test Loss', 'MAE', 'MSE', 'RMSE', 'SI', 'R-Value'])
    r_max = 0
    model_to_save = None
    for comb in combinations:
      # Evaluate the model with the selected and ranked features
      X_selected = X[:, comb]
      X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)

      # Standardize the features
      # scaler = StandardScaler()
      # X_train = scaler.fit_transform(X_train)
      # X_test = scaler.transform(X_test)

      # Train the MLP regressor
      mlp = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
      mlp.fit(X_train, y_train)
      y_pred = mlp.predict(X_test)

      # Calculate metrics
      mae = mean_absolute_error(y_test, y_pred)
      mse = mean_squared_error(y_test, y_pred)
      rmse = np.sqrt(mse)
      r_value = r2_score(y_test, y_pred)
      si = rmse / np.mean(y_test)
      test_loss = mlp.loss_
      if r_value > 0.1:
        pltloss(mlp)
        make_r_plot(y_test,y_pred)
        feature_important_names = [feature_names[f] for f in comb]

        # Append results using pd.concat
        new_row = pd.DataFrame({
            'Model':[str(combination_num) +'-'+ model_name],
            'Features': [feature_important_names],
            'Test Loss': [test_loss],
            'MAE': [mae],
            'MSE': [mse],
            'RMSE': [rmse],
            'SI': [si],
            'R-Value': [r_value]
        })
        if r_value > r_max:
          r_max = r_value
          model_to_save = new_row

        results = pd.concat([results, new_row], ignore_index=True)


        print("y_test: ", y_test)
        print("y_pred: ", y_pred)
    display(results)
    return results, model_to_save
  else:
    print("Number of combination is greater than the size of the ranked features")
    return None, None

"""# CLASS GREY WOLF OPTIMIZAION CODE

`This code implements a feature selection algorithm using the Grey Wolf Optimizer. It evaluates the fitness of each solution (wolf) by training an MLP regressor and calculating the mean squared error. The positions of the wolves are updated iteratively based on the social hierarchy and hunting behavior of grey wolves, with the aim of finding the subset of features that minimizes the MSE.`

```
num_features: Number of features in the dataset.
num_wolves: Number of wolves (solutions) in the population.
max_iter: Maximum number of iterations for the optimization process.
alpha_pos, beta_pos, delta_pos: Positions of the top three wolves (solutions) based on their fitness scores.
alpha_score, beta_score, delta_score: Fitness scores of the top three wolves.
population: Initial population of wolves, each represented by a binary vector indicating selected features.
fitness: Evaluates the fitness of a wolf (solution).
selected_features: Array of indices of selected features.
train_test_split: Splits the data into training and testing sets.
StandardScaler: Standardizes the features.
MLPRegressor: Trains an MLP regressor on the training data.
mean_squared_error: Calculates the mean squared error (MSE) between the actual and predicted labels.

update_position: Updates the position of a wolf based on the positions of the top three wolves (alpha, beta, delta).
a, A1, A2, A3, C1, C2, C3: Coefficients for updating the positions based on the current iteration.
D_alpha, D_beta, D_delta: Distances between the current wolf and the top three wolves.
X1, X2, X3: New positions calculated based on the distances.
```

Initialization:

> The GWO class initializes the population of wolves, their positions, and the alpha, beta, delta positions and scores.



Fitness Function:
> ANN to evaluate the fitness of feature subsets, leveraging the power of neural networks to potentially provide more accurate fitness evaluations for complex datasets.




Update Position:
> Wolves update their positions based on the positions of the alpha, beta, and delta wolves.


Binarization:
> Continuous position values are binarized to determine selected features.


Optimization Loop:
> The main loop of the GWO algorithm, iterating through each wolf and updating their positions based on their fitness. For each iteration, the fitness of each wolf is calculated, and the top three wolves (alpha, beta, delta) are updated based on their fitness scores. Positions of all wolves are updated based on the positions of the top three wolves. The binary positions of the best features (alpha wolf) are returned after the final iteration.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import warnings
import itertools
from sklearn.metrics import r2_score
from sklearn.exceptions import ConvergenceWarning

# Ignore convergence warnings for simplicity
warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")

# Define the Grey Wolf Optimizer class
class GWO:
    def __init__(self, num_features, num_wolves=5, max_iter=20):
        self.num_features = num_features
        self.num_wolves = num_wolves
        self.max_iter = max_iter
        # self.test_size = test_size
        self.alpha_pos = np.zeros(num_features)
        self.beta_pos = np.zeros(num_features)
        self.delta_pos = np.zeros(num_features)
        self.alpha_score = float("inf")
        self.beta_score = float("inf")
        self.delta_score = float("inf")
        self.population = np.random.randint(2, size=(num_wolves, num_features))

    def fitness(self, wolf, data, labels,test_size):
        selected_features = np.where(wolf == 1)[0]
        if selected_features.size == 0:
            return float("inf")
        X_selected = data[:, selected_features]
        X_train, X_test, y_train, y_test = train_test_split(X_selected, labels, test_size=test_size, random_state=42)

        # Standardize the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train the regressor
        regressor = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        return mse

    def update_position(self, wolf):
        a = 2 * (1 - (self.iteration / self.max_iter))
        r1, r2 = np.random.random(2)
        A1 = 2 * a * r1 - a
        C1 = 2 * r2
        D_alpha = abs(C1 * self.alpha_pos - wolf)
        X1 = self.alpha_pos - A1 * D_alpha

        r1, r2 = np.random.random(2)
        A2 = 2 * a * r1 - a
        C2 = 2 * r2
        D_beta = abs(C2 * self.beta_pos - wolf)
        X2 = self.beta_pos - A2 * D_beta

        r1, r2 = np.random.random(2)
        A3 = 2 * a * r1 - a
        C3 = 2 * r2
        D_delta = abs(C3 * self.delta_pos - wolf)
        X3 = self.delta_pos - A3 * D_delta

        new_wolf = (X1 + X2 + X3) / 3
        return np.clip(new_wolf, 0, 1)

    def binarize(self, wolf):
        return (wolf > 0.5).astype(int)

    def optimize(self, data, labels,test_size):
        for self.iteration in range(self.max_iter):
            for i in range(self.num_wolves):
                fitness = self.fitness(self.population[i], data, labels,test_size)
                if fitness < self.alpha_score:
                    self.delta_score = self.beta_score
                    self.delta_pos = self.beta_pos.copy()
                    self.beta_score = self.alpha_score
                    self.beta_pos = self.alpha_pos.copy()
                    self.alpha_score = fitness
                    self.alpha_pos = self.population[i].copy()
                elif fitness < self.beta_score:
                    self.delta_score = self.beta_score
                    self.delta_pos = self.beta_pos.copy()
                    self.beta_score = fitness
                    self.beta_pos = self.population[i].copy()
                elif fitness < self.delta_score:
                    self.delta_score = fitness
                    self.delta_pos = self.population[i].copy()

            for i in range(self.num_wolves):
                new_wolf = self.update_position(self.population[i])
                self.population[i] = self.binarize(new_wolf)

        return np.where(self.alpha_pos == 1)[0]

import pandas as pd
all_results = pd.DataFrame(columns=['Model','Features', 'Test Loss', 'MAE', 'MSE', 'RMSE', 'SI', 'R-Value'])

"""# 70-30 DATA SPLIT"""

# Load your dataset from an Excel file
# Assuming the last column is the label
X = data_all.iloc[:, :-1].values

y = data_all.iloc[:, -1].values

feature_names = data_all.columns.values.tolist()
# Ensure y is of numeric type
y = y.astype(float)

imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)
X1 = X
y1 = y
# Normalize the features

# X = scaler.fit_transform(X)
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

num_features = X.shape[1]
test_size=0.3
gwo = GWO(num_features, num_wolves=10, max_iter=30)
selected_features = gwo.optimize(X, y,test_size)
print("Selected features:", selected_features)

# Evaluate and rank the importance of each selected feature
feature_importance = {}
for feature in selected_features:
    X_selected = X[:, [feature]]
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train the MLP regressor
    mlp = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)

    # Calculate MSE for the feature
    mse = mean_squared_error(y_test, y_pred)
    feature_importance[feature] = mse

# Rank features by importance (MSE)
ranked_features = sorted(feature_importance, key=feature_importance.get)
ranked_features1 = ranked_features
print("Ranked features by importance:", ranked_features)

plotbar(ranked_features, feature_names)



# Initialize results DataFrame
results = pd.DataFrame(columns=['Model','Features', 'Test Loss', 'MAE', 'MSE', 'RMSE', 'SI', 'R-Value'])




# Evaluate the model with the selected and ranked features
X_selected = X[:, ranked_features]
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the MLP regressor
mlp = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)


pltloss(mlp)
make_r_plot(y_test,y_pred)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r_value = r2_score(y_test, y_pred)
si = rmse / np.mean(y_test)
test_loss = mlp.loss_

feature_important_names = [feature_names[f] for f in feature_importance]

# Append results using pd.concat
new_row = pd.DataFrame({
    'Model': ['70-30 Split'],
    'Features': [feature_important_names],
    'Test Loss': [test_loss],
    'MAE': [mae],
    'MSE': [mse],
    'RMSE': [rmse],
    'SI': [si],
    'R-Value': [r_value]
})
results = pd.concat([results, new_row], ignore_index=True)
all_results = pd.concat([all_results, new_row], ignore_index=True)

display(results)
print("y_test: ", y_test)
print("y_pred: ", y_pred)


# Print the performance metrics
print("\nModel Performance Metrics:")
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"SI: {si}")
print(f"R-Value: {r_value}")
print(f"Test Loss: {test_loss}")

"""## From the best input, combine two variables together........"""

pt, mds = combine_input_variables(2, ranked_features1,X1,y1,'70-30 model')
all_results = pd.concat([all_results, mds], ignore_index=True)

"""## From the best input, combine three variables together........"""

pt, mds = combine_input_variables(3, ranked_features1,X1,y1, '70-30 model')
all_results = pd.concat([all_results, mds], ignore_index=True)

"""## From the best input, combine four variables together........"""

pt, mds = combine_input_variables(4, ranked_features1,X1,y1, '70-30 model')
all_results = pd.concat([all_results, mds], ignore_index=True)

"""# 80-20 Split"""

# Load your dataset from an Excel file
# Assuming the last column is the label
X = data_all.iloc[:, :-1].values

y = data_all.iloc[:, -1].values

feature_names = data_all.columns.values.tolist()
# Ensure y is of numeric type
y = y.astype(float)

imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)
X2 = X
y2 = y
# Normalize the features

# X = scaler.fit_transform(X)
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

num_features = X.shape[1]
test_size = 0.2
gwo = GWO(num_features, num_wolves=10, max_iter=30)
selected_features = gwo.optimize(X, y,test_size)
print("Selected features:", selected_features)

# Evaluate and rank the importance of each selected feature
feature_importance = {}
for feature in selected_features:
    X_selected = X[:, [feature]]
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train the MLP regressor
    mlp = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)

    # Calculate MSE for the feature
    mse = mean_squared_error(y_test, y_pred)
    feature_importance[feature] = mse

# Rank features by importance (MSE)
ranked_features = sorted(feature_importance, key=feature_importance.get)
ranked_features2 = ranked_features
print("Ranked features by importance:", ranked_features)

plotbar(ranked_features, feature_names)



# Initialize results DataFrame
results = pd.DataFrame(columns=['Features', 'Test Loss', 'MAE', 'MSE', 'RMSE', 'SI', 'R-Value'])




# Evaluate the model with the selected and ranked features
X_selected = X[:, ranked_features]
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the MLP regressor
mlp = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)


pltloss(mlp)
make_r_plot(y_test,y_pred)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r_value = r2_score(y_test, y_pred)
si = rmse / np.mean(y_test)
test_loss = mlp.loss_

feature_important_names = [feature_names[f] for f in feature_importance]

# Append results using pd.concat
new_row = pd.DataFrame({
    'Model': ['80-20 Model'],
    'Features': [feature_important_names],
    'Test Loss': [test_loss],
    'MAE': [mae],
    'MSE': [mse],
    'RMSE': [rmse],
    'SI': [si],
    'R-Value': [r_value]
})
results = pd.concat([results, new_row], ignore_index=True)
all_results = pd.concat([all_results, new_row], ignore_index=True)

display(results)
print("y_test: ", y_test)
print("y_pred: ", y_pred)


# Print the performance metrics
print("\nModel Performance Metrics:")
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"SI: {si}")
print(f"R-Value: {r_value}")
print(f"Test Loss: {test_loss}")

# combine_input_variables(2, ranked_features2,X2,y2)
pt, mds = combine_input_variables(2, ranked_features2,X2,y2, '80-20 model')
all_results = pd.concat([all_results, mds], ignore_index=True)

pt, mds = combine_input_variables(3, ranked_features2,X2,y2,'80-20 model')
all_results = pd.concat([all_results, mds], ignore_index=True)

pt, mds = combine_input_variables(4, ranked_features2,X2,y2, '80-20 model')
all_results = pd.concat([all_results, mds], ignore_index=True)

"""# Feature engineering

```
Reprocess the data ans add extra features:
 'Year of construct',
 'location',
 'Duration of construction/Demolition (Month)',
 'Type of building',
 'Total Area of Building',
 'Number of floor',
 'Site access',
 'Area_Floor_Interaction',
 'Log_Total_Area',
 'Sqrt_Number_of_Floor',
 'Duration_Type_Interaction',
 'Location_Access_Interaction',
 'Amount of Waste (No. of Trucks)'
```


"""

import numpy as np

# Step 5.1: Feature Engineering

# Create new interaction features
data_cleaned['Area_Floor_Interaction'] = data_cleaned['Total Area of Building'] * data_cleaned['Number of floor']
data_cleaned['Log_Total_Area'] = np.log1p(data_cleaned['Total Area of Building'])
data_cleaned['Sqrt_Number_of_Floor'] = np.sqrt(data_cleaned['Number of floor'])
data_cleaned['Duration_Type_Interaction'] = data_cleaned['Duration of construction/Demolition (Month)'] * data_cleaned['Type of building']
data_cleaned['Location_Access_Interaction'] = data_cleaned['location'] * data_cleaned['Site access']
column_to_move = data_cleaned.pop('Amount of Waste (No. of Trucks)')
data_cleaned.insert(12, 'Amount of Waste (No. of Trucks)', column_to_move)

# Display the first few rows of the data with new features
data_cleaned.head()

# Load your dataset from an Excel file
# Assuming the last column is the label
X = data_cleaned.iloc[:, :-1].values
y = data_cleaned.iloc[:, -1].values

# Ensure y is of numeric type
y = y.astype(float)

imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)
X3 = X
y3 = y
# Normalize the features
scaler = StandardScaler()
# X = scaler.fit_transform(X)
# scaler = StandardScaler()
# X = scaler.fit_transform(X)
feature_names = data_cleaned.columns.values.tolist()
num_features = X.shape[1]
gwo = GWO(num_features, num_wolves=10, max_iter=30)
selected_features = gwo.optimize(X, y,0.2)
print("Selected features:", selected_features)

# Evaluate and rank the importance of each selected feature
feature_importance = {}
for feature in selected_features:
    X_selected = X[:, [feature]]
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train the MLP regressor
    mlp = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)

    # Calculate MSE for the feature
    mse = mean_squared_error(y_test, y_pred)
    feature_importance[feature] = mse

# Rank features by importance (MSE)
ranked_features = sorted(feature_importance, key=feature_importance.get)
print("Ranked features by importance:", ranked_features)

ranked_features3 = ranked_features

plotbar(ranked_features, feature_names)

# Initialize results DataFrame
results = pd.DataFrame(columns=['Features', 'Test Loss', 'MAE', 'MSE', 'RMSE', 'SI', 'R-Value'])




# Evaluate the model with the selected and ranked features
X_selected = X[:, ranked_features]
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the MLP regressor
mlp = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)


pltloss(mlp)
make_r_plot(y_test,y_pred)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r_value = r2_score(y_test, y_pred)
si = rmse / np.mean(y_test)
test_loss = mlp.loss_

feature_important_names = [feature_names[f] for f in feature_importance]

# Append results using pd.concat
new_row = pd.DataFrame({
    'Model':['Engineered Features'],
    'Features': [feature_important_names],
    'Test Loss': [test_loss],
    'MAE': [mae],
    'MSE': [mse],
    'RMSE': [rmse],
    'SI': [si],
    'R-Value': [r_value]
})
results = pd.concat([results, new_row], ignore_index=True)
all_results = pd.concat([all_results, new_row], ignore_index=True)
display(results)
print("y_test: ", y_test)
print("y_pred: ", y_pred)


# Print the performance metrics
print("\nModel Performance Metrics:")
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"SI: {si}")
print(f"R-Value: {r_value}")
print(f"Test Loss: {test_loss}")

"""## Combine two, three, four.... inputs"""

_, pts = combine_input_variables(2, ranked_features3,X3,y3,'Engineered model')
all_results = pd.concat([all_results, pts], ignore_index=True)

_,pts = combine_input_variables(3, ranked_features3,X3,y3, 'Engineered Model')
all_results = pd.concat([all_results, pts], ignore_index=True)

_, pts = combine_input_variables(3, ranked_features3 ,X3,y3,'Engineered Model')
results = pd.concat([results, pts], ignore_index=True)

"""# CROSS VALIDATION TASK

```
Cross-Validation in Fitness Calculation: The fitness method in the GWO class now uses cross_val_score with KFold cross-validation to calculate the mean MSE.


Cross-Validation for Final Evaluation: The final evaluation of the model with the selected features uses

cross_validate to obtain performance metrics from cross-validation.
Handling MSE as Test Loss: The mean MSE from cross-validation is used as the test loss for reporting.
```
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_validate
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
from sklearn.exceptions import ConvergenceWarning

# Ignore convergence warnings for simplicity
warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")

# Define the Grey Wolf Optimizer class
class GWO_crossval:
    def __init__(self, num_features, num_wolves=5, max_iter=20):
        self.num_features = num_features
        self.num_wolves = num_wolves
        self.max_iter = max_iter
        self.alpha_pos = np.zeros(num_features)
        self.beta_pos = np.zeros(num_features)
        self.delta_pos = np.zeros(num_features)
        self.alpha_score = float("inf")
        self.beta_score = float("inf")
        self.delta_score = float("inf")
        self.population = np.random.randint(2, size=(num_wolves, num_features))

    def fitness(self, wolf, data, labels):
        selected_features = np.where(wolf == 1)[0]
        if selected_features.size == 0:
            return float("inf")
        X_selected = data[:, selected_features]

        # Define the regressor
        regressor = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)

        # Use cross-validation to calculate the mean MSE
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        mse_scores = cross_val_score(regressor, X_selected, labels, cv=cv, scoring='neg_mean_squared_error')
        mean_mse = -mse_scores.mean()

        return mean_mse

    def update_position(self, wolf):
        a = 2 * (1 - (self.iteration / self.max_iter))
        r1, r2 = np.random.random(2)
        A1 = 2 * a * r1 - a
        C1 = 2 * r2
        D_alpha = abs(C1 * self.alpha_pos - wolf)
        X1 = self.alpha_pos - A1 * D_alpha

        r1, r2 = np.random.random(2)
        A2 = 2 * a * r1 - a
        C2 = 2 * r2
        D_beta = abs(C2 * self.beta_pos - wolf)
        X2 = self.beta_pos - A2 * D_beta

        r1, r2 = np.random.random(2)
        A3 = 2 * a * r1 - a
        C3 = 2 * r2
        D_delta = abs(C3 * self.delta_pos - wolf)
        X3 = self.delta_pos - A3 * D_delta

        new_wolf = (X1 + X2 + X3) / 3
        return np.clip(new_wolf, 0, 1)

    def binarize(self, wolf):
        return (wolf > 0.5).astype(int)

    def optimize(self, data, labels):
        for self.iteration in range(self.max_iter):
            for i in range(self.num_wolves):
                fitness = self.fitness(self.population[i], data, labels)
                if fitness < self.alpha_score:
                    self.delta_score = self.beta_score
                    self.delta_pos = self.beta_pos.copy()
                    self.beta_score = self.alpha_score
                    self.beta_pos = self.alpha_pos.copy()
                    self.alpha_score = fitness
                    self.alpha_pos = self.population[i].copy()
                elif fitness < self.beta_score:
                    self.delta_score = self.beta_score
                    self.delta_pos = self.beta_pos.copy()
                    self.beta_score = fitness
                    self.beta_pos = self.population[i].copy()
                elif fitness < self.delta_score:
                    self.delta_score = fitness
                    self.delta_pos = self.population[i].copy()

            for i in range(self.num_wolves):
                new_wolf = self.update_position(self.population[i])
                self.population[i] = self.binarize(new_wolf)

        return np.where(self.alpha_pos == 1)[0]

# Load your dataset from an Excel file


# Assuming the last column is the label
X = data_all.iloc[:, :-1].values

y = data_all.iloc[:, -1].values

feature_names = data_all.columns.values.tolist()

# Ensure y is of numeric type
y = y.astype(float)

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

num_features = X.shape[1]
gwo = GWO_crossval(num_features, num_wolves=10, max_iter=30)
selected_features = gwo.optimize(X, y)
print("Selected features:", selected_features)

# Evaluate and rank the importance of each selected feature
feature_importance = {}
for feature in selected_features:
    X_selected = X[:, [feature]]
    regressor = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_scores = cross_val_score(regressor, X_selected, y, cv=cv, scoring='neg_mean_squared_error')
    mean_mse = -mse_scores.mean()
    feature_importance[feature] = mean_mse

# Rank features by importance (MSE)
ranked_features = sorted(feature_importance, key=feature_importance.get)
print("Ranked features by importance:", ranked_features)

# Evaluate the model with the selected and ranked features using cross-validation
X_selected = X[:, ranked_features]
regressor = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)

cv_results = cross_validate(regressor, X_selected, y, cv=cv, scoring=('neg_mean_absolute_error', 'neg_mean_squared_error', 'r2'))
mae_scores = -cv_results['test_neg_mean_absolute_error']
mse_scores = -cv_results['test_neg_mean_squared_error']
r2_scores = cv_results['test_r2']

mean_mae = mae_scores.mean()
mean_mse = mse_scores.mean()
mean_rmse = np.sqrt(mean_mse)
mean_r2 = r2_scores.mean()
mean_si = mean_rmse / np.mean(y)
test_loss = mean_mse  # Using mean MSE as test loss
feature_important_names = [feature_names[f] for f in feature_importance]

new_row = pd.DataFrame({
    'Model': ['cross val Model'],
    'Features': [feature_important_names],
    'Test Loss': [test_loss],
    'MAE': [mean_mae],
    'MSE': [mean_mse],
    'RMSE': [mean_rmse],
    'SI': [mean_si],
    'R-Value': [mean_r2]
})
all_results = pd.concat([all_results, new_row], ignore_index=True)

# Print the performance metrics
print("\nModel Performance Metrics:")
print(f"MAE: {mean_mae}")
print(f"MSE: {mean_mse}")
print(f"RMSE: {mean_rmse}")
print(f"SI: {mean_si}")
print(f"R-Value: {mean_r2}")
print(f"Test Loss: {test_loss}")

# Plotting feature importance
plt.figure(figsize=(10, 6))
plt.bar(range(len(ranked_features)), [feature_importance[feature] for feature in ranked_features])
plt.xticks(range(len(ranked_features)), [feature_names[i] for i in ranked_features], rotation=45)
plt.xlabel('Features')
plt.ylabel('MSE')
plt.title('Feature Importance (MSE)')
plt.show()
regressor.fit(X_selected, y)
# Plotting the training loss curve
plt.figure(figsize=(10, 6))
plt.plot(regressor.loss_curve_)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.show()

"""# Tweaking Weights of Feature Importance

```
This code:
Normalizes the calculated importances to use them as weights.

Inverts the MSE scores to get higher weights for more important features (lower MSE).

Normalizes the weights so that their sum equals 1.
```
"""

# Load your dataset from an Excel file
# Assuming the last column is the label
X = data_all.iloc[:, :-1].values
y = data_all.iloc[:, -1].values

feature_names = data_all.columns.values.tolist()
# Ensure y is of numeric type
y = y.astype(float)

imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Normalize the features

# X = scaler.fit_transform(X)
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

num_features = X.shape[1]
test_size = 0.3
gwo = GWO(num_features, num_wolves=10, max_iter=30)
selected_features = gwo.optimize(X, y,test_size)
print("Selected features:", selected_features)

# Evaluate and rank the importance of each selected feature
feature_importance = {}
for feature in selected_features:
    X_selected = X[:, [feature]]
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train the MLP regressor
    mlp = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)

    # Calculate MSE for the feature
    mse = mean_squared_error(y_test, y_pred)
    feature_importance[feature] = mse

# Rank features by importance (MSE)
ranked_features = sorted(feature_importance, key=feature_importance.get)
ranked_features1 = ranked_features
print("Ranked features by importance:", ranked_features)

plotbar(ranked_features, feature_names)



# Initialize results DataFrame
results = pd.DataFrame(columns=['Features', 'Test Loss', 'MAE', 'MSE', 'RMSE', 'SI', 'R-Value'])


# we assign weights base on feature importance to each feature
# Compute feature weights
total_importance = sum(1 / (feature_importance[f] + 1e-6) for f in ranked_features)
feature_weights = {f: (1 / (feature_importance[f] + 1e-6)) / total_importance for f in ranked_features}

# Evaluate the model with the selected and ranked features with weights
X_selected = X[:, ranked_features]
X_weighted = X_selected.copy()
for i, feature in enumerate(ranked_features):
    X_weighted[:, i] *= feature_weights[feature]

X_train, X_test, y_train, y_test = train_test_split(X_weighted, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the MLP regressor
mlp = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)


pltloss(mlp)
make_r_plot(y_test,y_pred)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r_value = r2_score(y_test, y_pred)
si = rmse / np.mean(y_test)
test_loss = mlp.loss_

feature_important_names = [feature_names[f] for f in feature_importance]

# Append results using pd.concat
new_row = pd.DataFrame({
    'Model': ['Tweaking Weights of Features'],
    'Features': [feature_important_names],
    'Test Loss': [test_loss],
    'MAE': [mae],
    'MSE': [mse],
    'RMSE': [rmse],
    'SI': [si],
    'R-Value': [r_value]
})
results = pd.concat([results, new_row], ignore_index=True)
all_results = pd.concat([all_results, new_row], ignore_index=True)

display(results)
print("y_test: ", y_test)
print("y_pred: ", y_pred)


# Print the performance metrics
print("\nModel Performance Metrics:")
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"SI: {si}")
print(f"R-Value: {r_value}")
print(f"Test Loss: {test_loss}")

"""# MODEL SUMMARY"""

all_results
