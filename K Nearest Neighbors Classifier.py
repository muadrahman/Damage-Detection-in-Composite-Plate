#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score, f1_score, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score
import matplotlib.pyplot as plt


# In[2]:


# Load the dataset
dataset_path = "D:/MTP/Mid Term/MId Term 256 models.csv"
data = pd.read_csv(dataset_path)
data.head()


# In[3]:


# Extract features and target variables
X = data.iloc[:, 9:]  # Features: natural frequencies
y_horizontal = data['Model horizontal Class']
y_vertical = data['Model Vertical class']
y_box = data['Model Box  Class']


# In[4]:


def classify_knn(X, y, label):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the KNN classifier
    clf = KNeighborsClassifier()
    clf.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_train_pred = clf.predict(X_train_scaled)
    y_test_pred = clf.predict(X_test_scaled)
    
    return y_train, y_train_pred, y_test, y_test_pred


# In[5]:


# Perform classification for each target variable
train_actual_horizontal, train_predicted_horizontal, test_actual_horizontal, test_predicted_horizontal = classify_knn(X, y_horizontal, "Model Horizontal Class")
train_actual_vertical, train_predicted_vertical, test_actual_vertical, test_predicted_vertical = classify_knn(X, y_vertical, "Model Vertical Class")
train_actual_box, train_predicted_box, test_actual_box, test_predicted_box = classify_knn(X, y_box, "Model Box Class")


# In[6]:


train_results_df = pd.DataFrame({
    'Actual Horizontal Class': train_actual_horizontal,
    'Predicted Horizontal Class': train_predicted_horizontal,
    'Actual Vertical Class': train_actual_vertical,
    'Predicted Vertical Class': train_predicted_vertical,
    'Actual Box Class': train_actual_box,
    'Predicted Box Class': train_predicted_box
})

test_results_df = pd.DataFrame({
    'Actual Horizontal Class': test_actual_horizontal,
    'Predicted Horizontal Class': test_predicted_horizontal,
    'Actual Vertical Class': test_actual_vertical,
    'Predicted Vertical Class': test_predicted_vertical,
    'Actual Box Class': test_actual_box,
    'Predicted Box Class': test_predicted_box
}) 


# In[7]:


train_results_df['Horizontal Class Match'] = np.where(train_results_df['Actual Horizontal Class'] == train_results_df['Predicted Horizontal Class'], 'Yes', 'No')
train_results_df['Vertical Class Match'] = np.where(train_results_df['Actual Vertical Class'] == train_results_df['Predicted Vertical Class'], 'Yes', 'No')
train_results_df['Box Class Match'] = np.where(train_results_df['Actual Box Class'] == train_results_df['Predicted Box Class'], 'Yes', 'No')

test_results_df['Horizontal Class Match'] = np.where(test_results_df['Actual Horizontal Class'] == test_results_df['Predicted Horizontal Class'], 'Yes', 'No')
test_results_df['Vertical Class Match'] = np.where(test_results_df['Actual Vertical Class'] == test_results_df['Predicted Vertical Class'], 'Yes', 'No')
test_results_df['Box Class Match'] = np.where(test_results_df['Actual Box Class'] == test_results_df['Predicted Box Class'], 'Yes', 'No')

train_results_df['Cell Number'] = range(1, len(train_results_df) + 1)
test_results_df['Cell Number'] = range(len(train_results_df) + 1, len(train_results_df) + len(test_results_df) + 1)


# In[8]:


# Counting 'Yes' and 'No' in training results
train_yes_no_counts = {
    'Horizontal Class Match': train_results_df['Horizontal Class Match'].value_counts(),
    'Vertical Class Match': train_results_df['Vertical Class Match'].value_counts(),
    'Box Class Match': train_results_df['Box Class Match'].value_counts()
}

print("Training Results 'Yes' and 'No' Counts:")
for key, value in train_yes_no_counts.items():
    print(f"{key}:")
    print(f"Yes: {value['Yes']}, No: {value['No']}")
    print()


# In[9]:


# Counting 'Yes' and 'No' in testing results
test_yes_no_counts = {
    'Horizontal Class Match': test_results_df['Horizontal Class Match'].value_counts(),
    'Vertical Class Match': test_results_df['Vertical Class Match'].value_counts(),
    'Box Class Match': test_results_df['Box Class Match'].value_counts()
}

print("\nTesting Results 'Yes' and 'No' Counts:")
for key, value in test_yes_no_counts.items():
    print(f"{key}:")
    print(f"Yes: {value['Yes']}, No: {value['No']}")
    print()


# In[10]:


combined_results_df = pd.concat([train_results_df, test_results_df], ignore_index=True)


# In[11]:


def calculate_metrics_and_counts_df(actual, predicted):
    report = classification_report(actual, predicted, output_dict=True)
    counts = {'Yes': 0, 'No': 0}
    for key, value in report.items():
        if key not in ['accuracy', 'macro avg', 'weighted avg']:
            counts['Yes'] += value['precision'] * value['support']
            counts['No'] += (1 - value['precision']) * value['support']
    metrics_df = pd.DataFrame(report).transpose()
    metrics_df = metrics_df[metrics_df.index.isin(['0', '1'])]  # Consider only binary classes
    counts_df = pd.DataFrame([counts], index=['Counts'])
    metrics_counts_df = pd.concat([metrics_df, counts_df])
    return metrics_counts_df


# In[12]:


# Calculate metrics for entire dataset
accuracy_total_horizontal = accuracy_score(train_results_df['Actual Horizontal Class'], train_results_df['Predicted Horizontal Class'])
precision_total_horizontal = precision_score(train_results_df['Actual Horizontal Class'], train_results_df['Predicted Horizontal Class'], average='weighted')
recall_total_horizontal = recall_score(train_results_df['Actual Horizontal Class'], train_results_df['Predicted Horizontal Class'], average='weighted')
f1_total_horizontal = f1_score(train_results_df['Actual Horizontal Class'], train_results_df['Predicted Horizontal Class'], average='weighted')
confusion_matrix_total_horizontal = confusion_matrix(train_results_df['Actual Horizontal Class'], train_results_df['Predicted Horizontal Class'])

accuracy_total_vertical = accuracy_score(train_results_df['Actual Vertical Class'], train_results_df['Predicted Vertical Class'])
precision_total_vertical = precision_score(train_results_df['Actual Vertical Class'], train_results_df['Predicted Vertical Class'], average='weighted')
recall_total_vertical = recall_score(train_results_df['Actual Vertical Class'], train_results_df['Predicted Vertical Class'], average='weighted')
f1_total_vertical = f1_score(train_results_df['Actual Vertical Class'], train_results_df['Predicted Vertical Class'], average='weighted')
confusion_matrix_total_vertical = confusion_matrix(train_results_df['Actual Vertical Class'], train_results_df['Predicted Vertical Class'])

accuracy_total_box = accuracy_score(train_results_df['Actual Box Class'], train_results_df['Predicted Box Class'])
precision_total_box = precision_score(train_results_df['Actual Box Class'], train_results_df['Predicted Box Class'], average='weighted')
recall_total_box = recall_score(train_results_df['Actual Box Class'], train_results_df['Predicted Box Class'], average='weighted')
f1_total_box = f1_score(train_results_df['Actual Box Class'], train_results_df['Predicted Box Class'], average='weighted')
confusion_matrix_total_box = confusion_matrix(train_results_df['Actual Box Class'], train_results_df['Predicted Box Class'])


# In[13]:


# Print results for entire dataset - Horizontal Class
print("Metrics for Entire Dataset - Horizontal Class:")
print("Accuracy:", accuracy_total_horizontal)
print("Precision:", precision_total_horizontal)
print("Recall:", recall_total_horizontal)
print("F1 Score:", f1_total_horizontal)
print("Confusion Matrix:\n", confusion_matrix_total_horizontal)

# Print results for entire dataset - Vertical Class
print("\nMetrics for Entire Dataset - Vertical Class:")
print("Accuracy:", accuracy_total_vertical)
print("Precision:", precision_total_vertical)
print("Recall:", recall_total_vertical)
print("F1 Score:", f1_total_vertical)
print("Confusion Matrix:\n", confusion_matrix_total_vertical)

# Print results for entire dataset - Box Class
print("\nMetrics for Entire Dataset - Box Class:")
print("Accuracy:", accuracy_total_box)
print("Precision:", precision_total_box)
print("Recall:", recall_total_box)
print("F1 Score:", f1_total_box)
print("Confusion Matrix:\n", confusion_matrix_total_box)


# In[14]:


# Calculate metrics for training data
accuracy_train_horizontal = accuracy_score(train_results_df['Actual Horizontal Class'], train_results_df['Predicted Horizontal Class'])
precision_train_horizontal = precision_score(train_results_df['Actual Horizontal Class'], train_results_df['Predicted Horizontal Class'], average='weighted')
recall_train_horizontal = recall_score(train_results_df['Actual Horizontal Class'], train_results_df['Predicted Horizontal Class'], average='weighted')
f1_train_horizontal = f1_score(train_results_df['Actual Horizontal Class'], train_results_df['Predicted Horizontal Class'], average='weighted')
confusion_matrix_train_horizontal = confusion_matrix(train_results_df['Actual Horizontal Class'], train_results_df['Predicted Horizontal Class'])

accuracy_train_vertical = accuracy_score(train_results_df['Actual Vertical Class'], train_results_df['Predicted Vertical Class'])
precision_train_vertical = precision_score(train_results_df['Actual Vertical Class'], train_results_df['Predicted Vertical Class'], average='weighted')
recall_train_vertical = recall_score(train_results_df['Actual Vertical Class'], train_results_df['Predicted Vertical Class'], average='weighted')
f1_train_vertical = f1_score(train_results_df['Actual Vertical Class'], train_results_df['Predicted Vertical Class'], average='weighted')
confusion_matrix_train_vertical = confusion_matrix(train_results_df['Actual Vertical Class'], train_results_df['Predicted Vertical Class'])

accuracy_train_box = accuracy_score(train_results_df['Actual Box Class'], train_results_df['Predicted Box Class'])
precision_train_box = precision_score(train_results_df['Actual Box Class'], train_results_df['Predicted Box Class'], average='weighted')
recall_train_box = recall_score(train_results_df['Actual Box Class'], train_results_df['Predicted Box Class'], average='weighted')
f1_train_box = f1_score(train_results_df['Actual Box Class'], train_results_df['Predicted Box Class'], average='weighted')
confusion_matrix_train_box = confusion_matrix(train_results_df['Actual Box Class'], train_results_df['Predicted Box Class'])


# In[15]:


# Print results for training data - Horizontal Class
print("\nMetrics for Training Data - Horizontal Class:")
print("Accuracy:", accuracy_train_horizontal)
print("Precision:", precision_train_horizontal)
print("Recall:", recall_train_horizontal)
print("F1 Score:", f1_train_horizontal)
print("Confusion Matrix:\n", confusion_matrix_train_horizontal)

# Print results for training data - Vertical Class
print("\nMetrics for Training Data - Vertical Class:")
print("Accuracy:", accuracy_train_vertical)
print("Precision:", precision_train_vertical)
print("Recall:", recall_train_vertical)
print("F1 Score:", f1_train_vertical)
print("Confusion Matrix:\n", confusion_matrix_train_vertical)

# Print results for training data - Box Class
print("\nMetrics for Training Data - Box Class:")
print("Accuracy:", accuracy_train_box)
print("Precision:", precision_train_box)
print("Recall:", recall_train_box)
print("F1 Score:", f1_train_box)
print("Confusion Matrix:\n", confusion_matrix_train_box)


# In[16]:


# Calculate metrics for testing data
accuracy_test_horizontal = accuracy_score(test_results_df['Actual Horizontal Class'], test_results_df['Predicted Horizontal Class'])
precision_test_horizontal = precision_score(test_results_df['Actual Horizontal Class'], test_results_df['Predicted Horizontal Class'], average='weighted')
recall_test_horizontal = recall_score(test_results_df['Actual Horizontal Class'], test_results_df['Predicted Horizontal Class'], average='weighted')
f1_test_horizontal = f1_score(test_results_df['Actual Horizontal Class'], test_results_df['Predicted Horizontal Class'], average='weighted')
confusion_matrix_test_horizontal = confusion_matrix(test_results_df['Actual Horizontal Class'], test_results_df['Predicted Horizontal Class'])

accuracy_test_vertical = accuracy_score(test_results_df['Actual Vertical Class'], test_results_df['Predicted Vertical Class'])
precision_test_vertical = precision_score(test_results_df['Actual Vertical Class'], test_results_df['Predicted Vertical Class'], average='weighted')
recall_test_vertical = recall_score(test_results_df['Actual Vertical Class'], test_results_df['Predicted Vertical Class'], average='weighted')
f1_test_vertical = f1_score(test_results_df['Actual Vertical Class'], test_results_df['Predicted Vertical Class'], average='weighted')
confusion_matrix_test_vertical = confusion_matrix(test_results_df['Actual Vertical Class'], test_results_df['Predicted Vertical Class'])

accuracy_test_box = accuracy_score(test_results_df['Actual Box Class'], test_results_df['Predicted Box Class'])
precision_test_box = precision_score(test_results_df['Actual Box Class'], test_results_df['Predicted Box Class'], average='weighted')
recall_test_box = recall_score(test_results_df['Actual Box Class'], test_results_df['Predicted Box Class'], average='weighted')
f1_test_box = f1_score(test_results_df['Actual Box Class'], test_results_df['Predicted Box Class'], average='weighted')
confusion_matrix_test_box = confusion_matrix(test_results_df['Actual Box Class'], test_results_df['Predicted Box Class'])


# In[17]:


# Print results for testing data - Horizontal Class
print("\nMetrics for Testing Data - Horizontal Class:")
print("Accuracy:", accuracy_test_horizontal)
print("Precision:", precision_test_horizontal)
print("Recall:", recall_test_horizontal)
print("F1 Score:", f1_test_horizontal)
print("Confusion Matrix:\n", confusion_matrix_test_horizontal)

# Print results for testing data - Vertical Class
print("\nMetrics for Testing Data - Vertical Class:")
print("Accuracy:", accuracy_test_vertical)
print("Precision:", precision_test_vertical)
print("Recall:", recall_test_vertical)
print("F1 Score:", f1_test_vertical)
print("Confusion Matrix:\n", confusion_matrix_test_vertical)

# Print results for testing data - Box Class
print("\nMetrics for Testing Data - Box Class:")
print("Accuracy:", accuracy_test_box)
print("Precision:", precision_test_box)
print("Recall:", recall_test_box)
print("F1 Score:", f1_test_box)
print("Confusion Matrix:\n", confusion_matrix_test_box)


# In[18]:


excel_file_path = "D:/MTP/Second part ML project/combined_results_KNN.xlsx"
combined_results_df.to_excel(excel_file_path, index=False)

box_class_df = combined_results_df[['Actual Box Class','Predicted Box Class']]

random_20_rows = box_class_df.sample(n=20, random_state=42)

print("Random 20 rows from the box_class_df DataFrame:")
random_20_rows


# In[19]:


# Define the Damage_Plate grid size
plate_size = 3  # Adjusted for 20 alphabets

# Create a 3x3 matrix to represent the Damage_Plate grid
Damage_Plate_grid = np.zeros((plate_size, plate_size), dtype=int)

# Assign box numbers to each box in the grid
box_number = 1
for i in range(plate_size):
    for j in range(plate_size):
        Damage_Plate_grid[i, j] = box_number
        box_number += 1

# Function to mark alphabets and write box numbers in the center
def mark_alphabets(box_number_input, alphabet, color):
    # Find the row and column indices of the box
    for i in range(plate_size):
        for j in range(plate_size):
            if Damage_Plate_grid[i, j] == box_number_input:
                row_index = i
                col_index = j

    # Plot the alphabet randomly in the box
    rand_row = row_index + np.random.rand()
    rand_col = col_index + np.random.rand()
    ax.text(rand_col, rand_row, alphabet, fontsize=14, ha='center', va='center', color=color)
    
    # Write the box number in the center
    ax.text(col_index + 0.5, row_index + 0.5, str(box_number_input), fontsize=10, ha='center', va='center')

# Create the Damage_Plate grid plot
fig, ax = plt.subplots(figsize=(6, 6))

# Plot the Damage_Plate grid lines
for i in range(plate_size + 1):
    ax.axhline(y=i, color='black', linewidth=2)
    ax.axvline(x=i, color='black', linewidth=2)

# Get box numbers and corresponding values from your DataFrame (replace with your data)
box1_numbers = box_class_df['Actual Box Class'].sample(n=20, random_state=42).tolist()
box2_numbers = box_class_df['Predicted Box Class'].sample(n=20, random_state=42).tolist()
actual_values = [np.random.randint(1, 10) for _ in range(20)]  # Replace with actual values
predicted_values = [np.random.randint(1, 10) for _ in range(20)]  # Replace with predicted values

alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']
color1 = 'red'
color2 = 'blue'

# Mark alphabets and write box numbers
for i in range(20):
    mark_alphabets(box1_numbers[i], alphabets[i], color1)
    mark_alphabets(box2_numbers[i], alphabets[i], color2)

# Set plot limits and labels
ax.set_xlim(0, plate_size)
ax.set_ylim(0, plate_size)
ax.set_xticks(np.arange(0.5, plate_size, 1))
ax.set_yticks(np.arange(0.5, plate_size, 1))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.grid(False)

# Create text labels outside the plot
offset_x = 3.08  # Adjust x-offset for label placement
offset_y = 2.85  # Adjust y-offset for label placement

label_text1 = plt.text(offset_x, offset_y, "\u2022 Actual", ha='left', va='center', fontsize=12, color=color1)
label_text2 = plt.text(offset_x, offset_y - 0.1, "\u2022 Predicted", ha='left', va='center', fontsize=12, color=color2)

# Remove box around the labels (optional)
label_text1.set_bbox(dict(facecolor='none', edgecolor='none', pad=0))
label_text2.set_bbox(dict(facecolor='none', edgecolor='none', pad=0))

# Show the plot
plt.title('Damage Plate classified as 9 Boxes')
plt.show()


# In[20]:


# Selecting actual and predicted classes for the horizontal class
horizontal_class_df = combined_results_df[['Actual Horizontal Class', 'Predicted Horizontal Class']]

random_20_rows_horizontal = horizontal_class_df.sample(n=20)

print("Random 20 rows from the horizontal_class_df DataFrame:")
random_20_rows_horizontal


# In[21]:


# Extract the actual and predicted horizontal class from the DataFrame
actual_classes = random_20_rows_horizontal['Actual Horizontal Class'].tolist()[:10]
predicted_classes = random_20_rows_horizontal['Predicted Horizontal Class'].tolist()[:10]

# Define the layout of the plate and strip labels
strip_labels = ['5', '4', '3', '2', '1']  # Reverse order to label from bottom to top

# Create a new figure
fig, ax = plt.subplots(figsize=(8, 6))

# Plot horizontal strips and assign numbers
for i, label in enumerate(strip_labels):
    strip_y = (4 - i) / 5  # Adjusting the y-coordinate to represent bottom strip as 1
    rect_strip = plt.Rectangle((0, strip_y), 1, 1 / 5, fill=False, edgecolor='black', linewidth=1)
    ax.add_patch(rect_strip)
    # Add strip label at the left corner of each strip
    ax.text(0.02, strip_y + 0.02, label, ha='center', va='center', fontsize=12, color='black')

# Set axis limits and labels
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect('equal')

# Add title
plt.title('Damage Plate classify as 5 Horizontal Strips')

# Initialize lists to store input data for red and blue alphabets
red_alphabets = []
blue_alphabets = []

# Take 5 inputs for red alphabets
for i, strip_num in enumerate(actual_classes):
    letter = chr(ord('A') + i)  # Convert index to corresponding alphabet letter
    red_alphabets.append((letter, strip_num))

# Take 5 inputs for blue alphabets
for i, strip_num in enumerate(predicted_classes):
    letter = chr(ord('A') + i)  # Convert index to corresponding alphabet letter
    blue_alphabets.append((letter, strip_num))

# Function to mark alphabets on the plate
def mark_alphabets(alphabets, color):
    for letter, strip_num in alphabets:
        if 1 <= strip_num <= 5:
            strip_y = (strip_num - 1) / 5  # y-coordinate of the selected strip
            # Generate random x and y coordinates inside the selected strip
            random_x_coordinate = np.random.uniform(0.1, 0.9)  # Adjust the range as per the strip width
            random_y_coordinate = np.random.uniform(strip_y, strip_y + 0.2)  # Adjust the range as per the strip height
            # Marking the alphabet with specified color
            ax.text(random_x_coordinate, random_y_coordinate, letter, ha='center', va='center', fontsize=16, color=color)

# Create text labels outside the plot
offset_x = 1.08  # Adjust x-offset for label placement
offset_y = 0.9  # Adjust y-offset for label placement

label_text1 = plt.text(offset_x, offset_y, "\u2022 Actual", ha='left', va='center', fontsize=12, color=color1)
label_text2 = plt.text(offset_x, offset_y - 0.1, "\u2022 Predicted", ha='left', va='center', fontsize=12, color=color2)
    
# Mark red alphabets
mark_alphabets(red_alphabets, 'red')

# Mark blue alphabets
mark_alphabets(blue_alphabets, 'blue')

# Show the plot
plt.show()


# In[22]:


# Selecting actual and predicted classes for the vertical class
vertical_class_df = combined_results_df[['Actual Vertical Class', 'Predicted Vertical Class']]

# Select 20 random rows for demonstration
random_20_rows_vertical = vertical_class_df.sample(n=20)

print("Random 20 rows from the vertical_class_df DataFrame:")
random_20_rows_vertical


# In[23]:


# Extract the actual and predicted vertical class from the DataFrame
actual_classes = random_20_rows_vertical['Actual Vertical Class'].tolist()[:10]
predicted_classes = random_20_rows_vertical['Predicted Vertical Class'].tolist()[:10]

# Define the layout of the plate and strip labels
strip_labels = ['1', '2', '3', '4', '5']  # Label from bottom to top for vertical class

# Create a new figure
fig, ax = plt.subplots(figsize=(8, 6))

# Plot vertical strips and assign numbers
for i, label in enumerate(strip_labels):
    strip_x = i / 5  # Adjusting the x-coordinate to represent leftmost strip as 1
    rect_strip = plt.Rectangle((strip_x, 0), 1 / 5, 1, fill=False, edgecolor='black', linewidth=1)
    ax.add_patch(rect_strip)
    # Add strip label at the bottom corner of each strip
    ax.text(strip_x + 0.02, 0.02, label, ha='center', va='center', fontsize=12, color='black')

# Set axis limits and labels
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect('equal')

# Add title
plt.title('Damage Plate classify as 5 Vertical Strips')

# Initialize lists to store input data for red and blue alphabets
red_alphabets = []
blue_alphabets = []

# Take 5 inputs for red alphabets
for i, strip_num in enumerate(actual_classes):
    letter = chr(ord('A') + i)  # Convert index to corresponding alphabet letter
    red_alphabets.append((letter, strip_num))

# Take 5 inputs for blue alphabets
for i, strip_num in enumerate(predicted_classes):
    letter = chr(ord('A') + i)  # Convert index to corresponding alphabet letter
    blue_alphabets.append((letter, strip_num))

# Function to mark alphabets on the plate
def mark_alphabets(alphabets, color):
    for letter, strip_num in alphabets:
        if 1 <= strip_num <= 5:
            strip_x = (strip_num - 1) / 5  # x-coordinate of the selected strip
            # Generate random x and y coordinates inside the selected strip
            random_x_coordinate = np.random.uniform(strip_x, strip_x + 0.2)  # Adjust the range as per the strip width
            random_y_coordinate = np.random.uniform(0.1, 0.9)  # Adjust the range as per the strip height
            # Marking the alphabet with specified color
            ax.text(random_x_coordinate, random_y_coordinate, letter, ha='center', va='center', fontsize=16, color=color)

# Create text labels outside the plot
offset_x = 1.08  # Adjust x-offset for label placement
offset_y = .95  # Adjust y-offset for label placement

label_text1 = plt.text(offset_x, offset_y, "\u2022 Actual", ha='left', va='center', fontsize=12, color=color1)
label_text2 = plt.text(offset_x, offset_y - 0.1, "\u2022 Predicted", ha='left', va='center', fontsize=12, color=color2)

# Mark red alphabets
mark_alphabets(red_alphabets, 'red')

# Mark blue alphabets
mark_alphabets(blue_alphabets, 'blue')

# Show the plot
plt.show()

