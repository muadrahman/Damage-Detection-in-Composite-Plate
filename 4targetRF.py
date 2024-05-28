#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error, explained_variance_score, mean_squared_log_error
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random


# In[3]:


# Load dataset 
data = pd.read_csv("D:\MTP\Mid Term\MId Term 256 models.csv")
data


# In[3]:


# Remove specific columns using NumPy
columns_to_remove = ['Sl No', 'area', 'Model horizontal Class', 'Model Vertical class', 'Model Box  Class']
data.drop(columns=columns_to_remove, inplace=True)
data.head()


# In[4]:


# Select features and target variables
features = [' natural frequency of Mode 1', ' natural frequency of Mode 2', 
            ' natural frequency of Mode 3', ' natural frequency of Mode 4', 
            ' natural frequency of Mode 5', ' natural frequency of Mode 6']
targets = [' x1', ' y1', ' x2', ' y2'] 


# In[5]:


# Split the data into features (X) and targets (y)
X = data[features]
y = data[targets]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[6]:


# Initialize a multi-output regression model (RandomForestRegressor as an example)
base_regressor = RandomForestRegressor(random_state=42)

# Create a multi-output wrapper for the regressor
multi_output_regressor = MultiOutputRegressor(base_regressor)

# Define parameter grid
param_grid = {
    'estimator__n_estimators': [100, 200, 300],
    'estimator__max_depth': [None, 10, 20, 30],
    'estimator__min_samples_split': [2, 5, 10],
    'estimator__min_samples_leaf': [1, 2, 4]
}

# Initialize GridSearchCV with MultiOutputRegressor
grid_search = GridSearchCV(multi_output_regressor, param_grid, cv=5)

# Fit the grid search
grid_search.fit(X_train, y_train)

# Make predictions on the testing data
y_pred_test =grid_search.predict(X_test)

# Once the model is trained, predict values for the training set
y_pred_train = grid_search.predict(X_train)


# In[7]:


# Evaluate the model
mse = mean_squared_error(y_test, y_pred_test)
mae = mean_absolute_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)
medae = median_absolute_error(y_test, y_pred_test)
evs = explained_variance_score(y_test, y_pred_test)
msle = mean_squared_log_error(y_test, y_pred_test)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R-squared:", r2)
print("Median Absolute Error:", medae)
print("Explained Variance Score:", evs)
print("Mean Squared Logarithmic Error:", msle)


# In[8]:


#Create a DataFrame to store actual and predicted values
output_values = pd.DataFrame(columns=['Actual_x1', 'Actual_y1', 'Actual_x2', 'Actual_y2',
                                      'Predicted_x1', 'Predicted_y1', 'Predicted_x2', 'Predicted_y2'])

#Iterate through the training dataset and append actual and predicted values to the DataFrame
for i in range(len(y_train)):
    actual_x1_train = y_train.iloc[i][' x1']
    actual_y1_train = y_train.iloc[i][' y1']
    actual_x2_train = y_train.iloc[i][' x2']
    actual_y2_train = y_train.iloc[i][' y2']

    predicted_x1_train = y_pred_train[i][0]
    predicted_y1_train = y_pred_train[i][1]
    predicted_x2_train = y_pred_train[i][2]
    predicted_y2_train = y_pred_train[i][3]

    output_values = pd.concat([output_values, pd.DataFrame({'Actual_x1': [actual_x1_train],
                                                            'Actual_y1': [actual_y1_train],
                                                            'Actual_x2': [actual_x2_train],
                                                            'Actual_y2': [actual_y2_train],
                                                            'Predicted_x1': [predicted_x1_train],
                                                            'Predicted_y1': [predicted_y1_train],
                                                            'Predicted_x2': [predicted_x2_train],
                                                            'Predicted_y2': [predicted_y2_train]})],
                              ignore_index=True)

#Iterate through the testing dataset and append actual and predicted values to the DataFrame
for i in range(len(y_test)):
    actual_x1_test = y_test.iloc[i][' x1']
    actual_y1_test = y_test.iloc[i][' y1']
    actual_x2_test = y_test.iloc[i][' x2']
    actual_y2_test = y_test.iloc[i][' y2']

    predicted_x1_test = y_pred_test[i][0]
    predicted_y1_test = y_pred_test[i][1]
    predicted_x2_test = y_pred_test[i][2]
    predicted_y2_test = y_pred_test[i][3]

    output_values = pd.concat([output_values, pd.DataFrame({'Actual_x1': [actual_x1_test],
                                                            'Actual_y1': [actual_y1_test],
                                                            'Actual_x2': [actual_x2_test],
                                                            'Actual_y2': [actual_y2_test],
                                                            'Predicted_x1': [predicted_x1_test],
                                                            'Predicted_y1': [predicted_y1_test],
                                                            'Predicted_x2': [predicted_x2_test],
                                                            'Predicted_y2': [predicted_y2_test]})],
                              ignore_index=True)

# Display the DataFrame containing actual and predicted values
print("Combined Output Values:")
output_values


# In[9]:


# Calculate evaluation metrics for each target variable separately
metrics_dict = {}

for target in targets:
    # Select actual and predicted values for the target variable
    actual_values = output_values[f'Actual_{target.strip()}']  # Remove extra space using strip()
    predicted_values = output_values[f'Predicted_{target.strip()}']  # Remove extra space using strip()
    
    # Calculate evaluation metrics
    mse = mean_squared_error(actual_values, predicted_values)
    mae = mean_absolute_error(actual_values, predicted_values)
    r2 = r2_score(actual_values, predicted_values)
    medae = median_absolute_error(actual_values, predicted_values)
    evs = explained_variance_score(actual_values, predicted_values)
    msle = mean_squared_log_error(actual_values, predicted_values)
    
    # Store metrics in a dictionary
    metrics_dict[target] = {
        'Mean Squared Error': mse,
        'Mean Absolute Error': mae,
        'R-squared': r2,
        'Median Absolute Error': medae,
        'Explained Variance Score': evs,
        'Mean Squared Logarithmic Error': msle
    }

# Print metrics for each target variable
for target, metrics in metrics_dict.items():
    print(f"Metrics for {target}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
    print()


# In[10]:


# Specify the file path
file_path = r'D:\CV things\ML projects\MTP projects\combined_output_values.csv'

# Write the combined data to the specified file path
output_values.to_csv(file_path, index=False)

print(f"Combined data saved to '{file_path}'")


# In[11]:


# Read the CSV file into a DataFrame
file_path = r'D:\CV things\ML projects\MTP projects\combined_output_values.csv'
output_values = pd.read_csv(file_path)

# Set the style for the plot
sns.set_style("whitegrid")

# Create a line plot for actual and predicted x1 values
plt.figure(figsize=(10, 6))

# Plot actual x1 values in blue
sns.lineplot(data=output_values['Actual_x1'], marker='o', color='blue', markersize=8, linestyle='-', label='Actual x1')

# Plot predicted x1 values in red
sns.lineplot(data=output_values['Predicted_x1'], marker='o', color='red', markersize=8, linestyle='-', label='Predicted x1')

# Add labels and title
plt.xlabel('Sample')
plt.ylabel('Value')
plt.title('Actual x1 Values vs Predicted x1 Values')

# Add legend
plt.legend()

# Show the plot
plt.show()


# In[12]:


# Set the palette to 'bright'
sns.set_palette('deep')

# Create a line plot for actual and predicted y1 values
plt.figure(figsize=(10, 6))

# Plot actual y1 values in blue
sns.lineplot(data=output_values['Actual_y1'], marker='o', color='skyblue', markersize=8, linestyle='-' ,label='Actual y1')

# Plot predicted y1 values in red
sns.lineplot(data=output_values['Predicted_y1'], marker='o', color='purple', markersize=8, linestyle='-', label='Predicted y1')

# Add labels and title
plt.xlabel('Sample')
plt.ylabel('Value')
plt.title('Actual y1 Values vs Predicted y1 Values')

# Add legend
plt.legend()

# Show the plot
plt.show()


# In[13]:


# Create a line plot for actual and predicted x2 values
plt.figure(figsize=(10, 6))

# Plot actual x2 values in green
sns.lineplot(data=output_values['Actual_x2'], marker='o', color='yellow', markersize=8, linestyle='-' ,label='Actual x2')

# Plot predicted x2 values in orange
sns.lineplot(data=output_values['Predicted_x2'], marker='o', color='green', markersize=8, linestyle='-', label='Predicted x2')

# Add labels and title
plt.xlabel('Sample')
plt.ylabel('Value')
plt.title('Actual vs Predicted x2 Values')

# Add legend
plt.legend()

# Show the plot
plt.show()


# In[14]:


# Create a line plot for actual and predicted y2 values
plt.figure(figsize=(10, 6))

# Plot actual y2 values in purple
sns.lineplot(data=output_values['Actual_y2'], marker='o', color='violet', markersize=8, linestyle='-' ,label='Actual y2')

# Plot predicted y2 values in yellow
sns.lineplot(data=output_values['Predicted_y2'], marker='o', color='black', markersize=8, linestyle='-', label='Predicted y2')

# Add labels and title
plt.xlabel('Sample')
plt.ylabel('Value')
plt.title('Actual y2 Values vs Predicted y2 Values')

# Add legend
plt.legend()

# Show the plot
plt.show()


# In[15]:


# Randomly select 100 rows from the DataFrame
sampled_data = output_values.sample(n=15)

# Create a scatter plot for actual and predicted x1 values
plt.figure(figsize=(10, 6))

# Plot actual x1 values in blue
plt.scatter(sampled_data.index, sampled_data['Actual_x1'], color='blue', label='Actual x1')

# Plot predicted x1 values in red
plt.scatter(sampled_data.index, sampled_data['Predicted_x1'], color='red', label='Predicted x1')

# Add labels and title
plt.xlabel('Sample')
plt.ylabel('Value')
plt.title('Actual x1 Values vs Predicted x1 Values - Scatter Plot')

# Add legend
plt.legend()

# Show the plot
plt.show()


# In[16]:


# Randomly select 15 rows from the DataFrame
sampled_data = output_values.sample(n=15, random_state=40)

# Create a scatter plot for actual and predicted y1 values
plt.figure(figsize=(10, 6))

# Plot actual y1 values in blue
plt.scatter(sampled_data.index, sampled_data['Actual_y1'], color='skyblue', label='Actual y1')

# Plot predicted y1 values in red
plt.scatter(sampled_data.index, sampled_data['Predicted_y1'], color='purple', label='Predicted y1')

# Add labels and title
plt.xlabel('Sample')
plt.ylabel('Value')
plt.title('Actual y1 Values vs Predicted y1 Values - Scatter Plot')

# Add legend
plt.legend()

# Show the plot
plt.show()


# In[17]:


# Create a scatter plot for actual and predicted x2 values
plt.figure(figsize=(10, 6))

# Plot actual x2 values in green
plt.scatter(sampled_data.index, sampled_data['Actual_x2'], color='yellow', label='Actual x2')

# Plot predicted x2 values in orange
plt.scatter(sampled_data.index, sampled_data['Predicted_x2'], color='green', label='Predicted x2')

# Add labels and title
plt.xlabel('Sample')
plt.ylabel('Value')
plt.title('Actual x2 Values vs Predicted x2 Values - Scatter Plot')

# Add legend
plt.legend()

# Show the plot
plt.show()


# In[18]:


# Create a scatter plot for actual and predicted y2 values
plt.figure(figsize=(10, 6))

# Plot actual y2 values in purple
plt.scatter(sampled_data.index, sampled_data['Actual_y2'], color='violet', label='Actual y2')

# Plot predicted y2 values in yellow
plt.scatter(sampled_data.index, sampled_data['Predicted_y2'], color='black', label='Predicted y2')

# Add labels and title
plt.xlabel('Sample')
plt.ylabel('Value')
plt.title('Actual y2 Values vs Predicted y2 Values - Scatter Plot')

# Add legend
plt.legend()

# Show the plot
plt.show()


# In[21]:


# Read the sample data from the CSV file
file_path = r'D:\CV things\ML projects\MTP projects\combined_output_values.csv'
data = pd.read_csv(file_path)

# Select 5 random rows from the data
random_indices = random.sample(range(len(data)),15)
random_data = data.iloc[random_indices]

# Extracting data from the DataFrame
for i, row in random_data.iterrows():
    actual_x1, actual_y1, actual_x2, actual_y2 = row[['Actual_x1', 'Actual_y1', 'Actual_x2', 'Actual_y2']]
    predicted_x1, predicted_y1, predicted_x2, predicted_y2 = row[['Predicted_x1', 'Predicted_y1', 'Predicted_x2', 'Predicted_y2']]

    # Plot rectangles
    plt.plot([actual_x1, actual_x2], [actual_y1, actual_y1], color='red')  # Bottom line
    plt.plot([actual_x1, actual_x2], [actual_y2, actual_y2], color='red')  # Top line
    plt.plot([actual_x1, actual_x1], [actual_y1, actual_y2], color='red')  # Left line
    plt.plot([actual_x2, actual_x2], [actual_y1, actual_y2], color='red')  # Right line

    plt.plot([predicted_x1, predicted_x2], [predicted_y1, predicted_y1], color='blue')  # Bottom line
    plt.plot([predicted_x1, predicted_x2], [predicted_y2, predicted_y2], color='blue')  # Top line
    plt.plot([predicted_x1, predicted_x1], [predicted_y1, predicted_y2], color='blue')  # Left line
    plt.plot([predicted_x2, predicted_x2], [predicted_y1, predicted_y2], color='blue')  # Right line
 
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Randomly Selected Actual vs Predicted Data with Rectangles')

#Show the plot
plt.show()

