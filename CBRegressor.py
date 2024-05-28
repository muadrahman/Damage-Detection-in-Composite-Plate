#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


# Load dataset 
data = pd.read_csv("D:\MTP\Mid Term\MId Term 256 models.csv")
data


# In[3]:


# Select features and target variables
features = [' natural frequency of Mode 1', ' natural frequency of Mode 2', 
            ' natural frequency of Mode 3', ' natural frequency of Mode 4', 
            ' natural frequency of Mode 5', ' natural frequency of Mode 6']
targets = [' x1', ' y1', ' x2', ' y2']


# In[4]:


# Initialize dictionaries to store actual and predicted values for both training and test sets
actual_predicted_train = {}
actual_predicted_test = {}
ensemble_models = {}

# Train CatBoost models for each target variable
for target in targets:
    y = data[target]
    X = data[features]  # Define features
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize CatBoostRegressor
    model = CatBoostRegressor(iterations=1000,
                              learning_rate=0.1,
                              depth=6,
                              loss_function='RMSE',
                              verbose=100)

    # Fit the model
    model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True, verbose=False)

    # Save the actual and predicted values for the training set
    y_pred_train = model.predict(X_train)
    actual_predicted_train[target] = (y_train.values, y_pred_train)

    # Save the actual and predicted values for the test set
    y_pred_test = model.predict(X_test)
    actual_predicted_test[target] = (y_test.values, y_pred_test)

    # Evaluate the model on the training set
    mse_train, mae_train, r2_train = mean_squared_error(y_train, y_pred_train), mean_absolute_error(y_train, y_pred_train), r2_score(y_train, y_pred_train)

    print(f"Metrics for CatBoost model on {target} (Train set):")
    print("Mean Squared Error:", mse_train)
    print("Mean Absolute Error:", mae_train)
    print("R-squared:", r2_train)
    print()

    # Evaluate the model on the test set
    mse_test, mae_test, r2_test = mean_squared_error(y_test, y_pred_test), mean_absolute_error(y_test, y_pred_test), r2_score(y_test, y_pred_test)

    print(f"Metrics for CatBoost model on {target} (Test set):")
    print("Mean Squared Error:", mse_test)
    print("Mean Absolute Error:", mae_test)
    print("R-squared:", r2_test)
    print()

    # Store the model in the ensemble
    ensemble_models[target] = model


# In[5]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Initialize dictionaries to store actual and predicted values for the entire dataset
actual_predicted_all = {}

# Aggregate actual and predicted values for all target variables
for target in targets:
    # Combine actual and predicted values for the training set
    y_train_actual, y_train_predicted = actual_predicted_train[target]
    # Combine actual and predicted values for the test set
    y_test_actual, y_test_predicted = actual_predicted_test[target]
    
    # Combine actual and predicted values for the entire dataset
    y_actual = np.concatenate([y_train_actual, y_test_actual])
    y_predicted = np.concatenate([y_train_predicted, y_test_predicted])
    
    # Store the aggregated actual and predicted values
    actual_predicted_all[target] = (y_actual, y_predicted)

# Calculate evaluation metrics for the entire dataset
for target, (y_actual, y_predicted) in actual_predicted_all.items():
    mse = mean_squared_error(y_actual, y_predicted)
    mae = mean_absolute_error(y_actual, y_predicted)
    r2 = r2_score(y_actual, y_predicted)
    
    print(f"Metrics for entire dataset ({target}):")
    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)
    print("R-squared:", r2)
    print()


# In[6]:


# Combine actual and predicted values for training set into a DataFrame
train_dataframes = []
for target, (y_actual, y_pred) in actual_predicted_train.items():
    train_df = pd.DataFrame({f'Actual_{target}': y_actual, f'Predicted_{target}': y_pred})
    train_dataframes.append(train_df)


# In[7]:


# Combine actual and predicted values for test set into a DataFrame
test_dataframes = []
for target, (y_actual, y_pred) in actual_predicted_test.items():
    test_df = pd.DataFrame({f'Actual_{target}': y_actual, f'Predicted_{target}': y_pred})
    test_dataframes.append(test_df)


# In[8]:


# Concatenate DataFrames for both training and test sets
combined_df = pd.concat([pd.concat(train_dataframes, axis=1), pd.concat(test_dataframes, axis=1)], ignore_index=True)

# Add the 'Sl No' column from the original dataset
combined_df.insert(0, 'Sl No', data['Sl No'])


# In[9]:


combined_df


# In[10]:


# Save the sorted combined DataFrame to an Excel file
output_file_path = r"D:\MTP\coding\last_output.xlsx"
combined_df.to_excel(output_file_path, index=False)

print(f"Last output saved to: {output_file_path}")


# In[11]:


# Randomly select 100 rows from the DataFrame
sampled_data = combined_df.sample(n=15, random_state=40)

# Create a scatter plot for actual and predicted x1 values
plt.figure(figsize=(10, 6))

# Plot actual x1 values in blue
plt.scatter(sampled_data.index, sampled_data['Actual_ x1'], color='blue', marker='s', s=150, label='Actual x1')

# Plot predicted x1 values in red
plt.scatter(sampled_data.index, sampled_data['Predicted_ x1'], color='red', marker='o', s=100, label='Predicted x1')

# Add labels and title
plt.xlabel('Sample')
plt.ylabel('Value')
plt.title('Actual vs Predicted x1 Values (Sampled) - Scatter Plot')

# Add legend
plt.legend()

# Show the plot
plt.show()


# In[12]:


# Create a scatter plot for actual and predicted y1 values
plt.figure(figsize=(10, 6))

# Plot actual y1 values in green
plt.scatter(sampled_data.index, sampled_data['Actual_ y1'], color='cyan', marker='s', s=150, label='Actual y1')

# Plot predicted y1 values in purple
plt.scatter(sampled_data.index, sampled_data['Predicted_ y1'], color='purple', marker='o', s=100, label='Predicted y1')

# Add labels and title
plt.xlabel('Sample')
plt.ylabel('Value')
plt.title('Actual vs Predicted y1 Values (Sampled) - Scatter Plot')

# Add legend
plt.legend()

# Show the plot
plt.show()


# In[13]:


# Create a scatter plot for actual and predicted x2 values
plt.figure(figsize=(10, 6))

# Plot actual x2 values in orange
plt.scatter(sampled_data.index, sampled_data['Actual_ x2'], color='orange', marker='s', s=150, label='Actual x2')

# Plot predicted x2 values in cyan
plt.scatter(sampled_data.index, sampled_data['Predicted_ x2'], color='green', marker='o', s=100, label='Predicted x2')

# Add labels and title
plt.xlabel('Sample')
plt.ylabel('Value')
plt.title('Actual vs Predicted x2 Values (Sampled) - Scatter Plot')

# Add legend
plt.legend()

# Show the plot
plt.show()


# In[14]:


# Create a scatter plot for actual and predicted y2 values
plt.figure(figsize=(10, 6))

# Plot actual y2 values in magenta
plt.scatter(sampled_data.index, sampled_data['Actual_ y2'], color='magenta', marker='s', s=150, label='Actual y2')

# Plot predicted y2 values in yellow
plt.scatter(sampled_data.index, sampled_data['Predicted_ y2'], color='yellow', marker='o', s=100, label='Predicted y2')

# Add labels and title
plt.xlabel('Sample')
plt.ylabel('Value')
plt.title('Actual vs Predicted y2 Values (Sampled) - Scatter Plot')

# Add legend
plt.legend()

# Show the plot
plt.show()


# In[27]:


import matplotlib.pyplot as plt

# Sampled data containing actual and predicted values
sampled_data = combined_df.sample(n=20)

# Create a figure and axis object
fig, ax = plt.subplots(figsize=(10, 6))

# Plot actual squares
for i, row in sampled_data.iterrows():
    actual_x1, actual_y1 = row['Actual_ x1'], row['Actual_ y1']
    actual_x2, actual_y2 = row['Actual_ x2'], row['Actual_ y2']
    
    # Plot actual square with a solid line
    ax.plot([actual_x1, actual_x2], [actual_y1, actual_y1], color='blue', linestyle='solid', linewidth=2)  # Bottom line
    ax.plot([actual_x1, actual_x2], [actual_y2, actual_y2], color='blue', linestyle='solid', linewidth=2)  # Top line
    ax.plot([actual_x1, actual_x1], [actual_y1, actual_y2], color='blue', linestyle='solid', linewidth=2)  # Left line
    ax.plot([actual_x2, actual_x2], [actual_y1, actual_y2], color='blue', linestyle='solid', linewidth=2)  # Right line

# Plot predicted squares
for i, row in sampled_data.iterrows():
    predicted_x1, predicted_y1 = row['Predicted_ x1'], row['Predicted_ y1']
    predicted_x2, predicted_y2 = row['Predicted_ x2'], row['Predicted_ y2']
    
    # Plot predicted square with a dashed line
    ax.plot([predicted_x1, predicted_x2], [predicted_y1, predicted_y1], color='red', linestyle='dashed', linewidth=2)  # Bottom line
    ax.plot([predicted_x1, predicted_x2], [predicted_y2, predicted_y2], color='red', linestyle='dashed', linewidth=2)  # Top line
    ax.plot([predicted_x1, predicted_x1], [predicted_y1, predicted_y2], color='red', linestyle='dashed', linewidth=2)  # Left line
    ax.plot([predicted_x2, predicted_x2], [predicted_y1, predicted_y2], color='red', linestyle='dashed', linewidth=2)  # Right line

# Add labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Actual vs Predicted Squares')

# Add labels indicating actual and predicted values
ax.text(1.15, 0.9, 'Actual Values', verticalalignment='top', horizontalalignment='right', transform=ax.transAxes, color='blue', fontsize=12)
ax.text(1.18, 0.85, 'Predicted Values', verticalalignment='top', horizontalalignment='right', transform=ax.transAxes, color='red', fontsize=12)
# Show the plot
plt.show()


# In[ ]:




