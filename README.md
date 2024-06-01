# Damage Detection in Composite Plates

Welcome to the repository for our innovative research project on damage detection in Glass Fiber Reinforced Polymer (GFRP) composite plates using advanced machine learning techniques. This project is part of the Master of Technology program in Civil Engineering with a specialization in Structural Engineering at the prestigious Indian Institute of Technology Kharagpur. It marks a significant advancement in structural health monitoring.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Key Components](#key-components)
3. [Project Workflow](#project-workflow)
4. [Key Findings and Results](#key-findings-and-results)
   - [Classification Algorithms for Damage Detection](#classification-algorithms-for-damage-detection)
   - [Regression Algorithms for Damage Severity Prediction](#regression-algorithms-for-damage-severity-prediction)
5. [Conclusions and Scope for Future Work](#conclusions-and-scope-for-future-work)
   - [Conclusions](#conclusions)
   - [Limitations](#limitations)
   - [Scope for Future Work](#scope-for-future-work)
6. [Contact Information](#contact-information)
7. [Resources](#resources)

## Project Overview

### Objective

Transform traditional damage detection methods by incorporating state-of-the-art machine learning algorithms to automate and improve the efficiency of damage detection processes in GFRP composite plates.

## Key Components

### Data Collection and Preprocessing

- **Data Collection**: Extensive data on the structural response of GFRP composite plates was collected using experimental modal testing and finite element simulations.
- **Preprocessing**: Python scripting was utilized for preprocessing raw data and extracting relevant features for machine learning models.

### Classification Algorithms for Damage Detection

- **K-Nearest Neighbors (KNN)**: Implemented to classify damage within predefined regions of the composite plates.
- **Support Vector Machine (SVM)**: Used for damage classification with higher precision and recall metrics.

### Regression Algorithms for Damage Severity Prediction

- **Random Forest**: Applied to predict damage severity and coordinates within the composite plates.
- **CatBoost**: Used for regression tasks, excelling in handling categorical features and capturing complex data relationships.

## Project Workflow

1. **Data Collection**: Gathered comprehensive data on natural frequencies and structural responses through experimental modal testing and finite element simulations.
2. **Data Preprocessing**: Processed raw data to extract relevant features and target variables for machine learning models.
3. **Model Training**: Trained classification and regression algorithms using the preprocessed data to develop predictive models for damage detection and severity assessment.
4. **Model Evaluation**: Assessed model performance using metrics such as accuracy, precision, recall, mean squared error (MSE), and R-squared score.
5. **Result Analysis**: Analyzed results from different algorithms to identify strengths, weaknesses, and areas for improvement in damage detection methodologies.

## Key Findings and Results

### Classification Algorithms for Damage Detection

#### K-Nearest Neighbors (KNN)

The provided code performs classification using the K-Nearest Neighbors (KNN) algorithm on a dataset containing features related to natural frequencies and several target variables. Here's a detailed summary covering each aspect:

## Data Preprocessing

- **Loading Data**: The dataset is loaded from a CSV file using pandas' `read_csv` function.
- **Feature Extraction**: Features (natural frequencies) and target variables (horizontal, vertical, and box classes) are extracted from the dataset.

## Model Training and Evaluation

- **KNN Classification Function**: A function `classify_knn` is defined to perform KNN classification. It splits the data into training and testing sets, standardizes the features using `StandardScaler`, trains the KNN classifier, and makes predictions.
- **Classification for Each Target Variable**: The `classify_knn` function is called separately for each target variable (horizontal class, vertical class, and box class).

## Train-Test Split and Standardization

- **Train-Test Split**: The dataset is split into training and testing sets using the `train_test_split` function from scikit-learn.
- **Standardization**: Features are standardized using `StandardScaler` to have a mean of 0 and a standard deviation of 1.

## Classification Report

- **Classification Report**: The classification report is generated using `classification_report` from scikit-learn, providing a comprehensive summary of precision, recall, F1-score, and support for each class.

## Management of Output Values

- **Results DataFrames**: Separate DataFrames are created to store the actual and predicted values for training and testing data for each target variable.
- **Matching Indicators**: Additional columns are added to indicate whether the predicted classes match the actual classes for each target variable.
- **Counting Matches**: Counts of 'Yes' and 'No' for matching between actual and predicted classes are calculated for both training and testing data.

## Visualization

- **Visualization of Results**: The confusion matrices are visualized to understand the performance of the classifiers visually.


## Evaluation Metrics

- **Model Training**: A K-Nearest Neighbors (KNN) classifier is trained for each target variable using the training data.
- **Prediction**: Predictions are made on both training and testing data using the trained models.
- **Metrics Calculation**: Various evaluation metrics such as accuracy, precision, recall, F1-score, and confusion matrix are calculated for both training and testing data for each target variable.


- **Horizontal Class**
  - Accuracy: 79.90%
  - Precision: 80.50%
  - Recall: 79.90%
  - F1 Score: 79.90%

- **Vertical Class**
  - Accuracy: 81.37%
  - Precision: 81.30%
  - Recall: 81.37%
  - F1 Score: 81.20%

- **Box Class**
  - Accuracy: 79.41%
  - Precision: 80.80%
  - Recall: 79.41%
  - F1 Score: 79.41%

#### Support Vector Machine (SVM)

The provided code conducts a supervised machine learning task using Support Vector Machines (SVMs) to classify data into multiple classes based on features extracted from a dataset. Here's a detailed summary covering various aspects of the code:

## 1. Data Loading and Preprocessing:

- **Loading Data:** The dataset is loaded from a CSV file using pandas.
- **Feature Extraction:** Features (X) and target variables (y) are extracted from the dataset.
- **Features and Targets:** Features consist of natural frequencies, while target variables are categorical classes.

## 2. Model Training:

- **Train-Test Split:** The dataset is split into training and testing sets using an 80-20 split ratio.
- **Feature Scaling:** Feature scaling is applied using StandardScaler to standardize features by removing the mean and scaling to unit variance.
- **Feature Selection:** Feature selection is performed using SelectKBest with ANOVA F-value to select the top k features.
- **Hyperparameter Tuning:** GridSearchCV is utilized for hyperparameter tuning, exploring different combinations of parameters (C, gamma, kernel) for the SVM classifier.
- **Model Fitting:** The trained SVM model is fitted using the training data.

## 3. Model Evaluation:

- **Classification Reports:** Classification reports are generated for both training and testing data, providing metrics such as precision, recall, F1-score, and support for each class.
- **Confusion Matrices:** Confusion matrices are computed to visualize the performance of the classifier across different classes.


## 5. Visualization:

- **Confusion Matrix Visualization:** Although the code does not include explicit visualization methods, confusion matrices can be visualized using libraries like Matplotlib for a better understanding of the classifier's performance.

# Summary of Key Findings:

- The SVM classifier achieves high accuracy and overall good performance in classifying the data into multiple classes based on natural frequencies.
- The classification performance varies slightly across different target variables (horizontal, vertical, and box classes).
- Precision, recall, and F1-score provide insights into the classifier's performance for each individual class.
- Confusion matrices reveal the classification errors and the model's confusion between different classes.

## 4. Results and Analysis:

- **Printing Metrics:** The code prints out various metrics for each target variable (horizontal class, vertical class, box class) separately for both training and testing data.
- **Metrics:** Metrics include accuracy, precision, recall, and F1-score, along with confusion matrices.
- **Analysis:** The results help in assessing the performance of the SVM classifier in classifying the data into different classes.


- **Horizontal Class**
  - Accuracy: 87.70%
  - Precision: 89.80%
  - Recall: 87.70%
  - F1 Score: 87.90%

- **Vertical Class**
  - Accuracy: 86.80%
  - Precision: 89.80%
  - Recall: 86.80%
  - F1 Score: 87.10%

- **Box Class**
  - Accuracy: 95.60%
  - Precision: 96.10%
  - Recall: 95.60%
  - F1 Score: 95.70%

### Regression Algorithms for Damage Severity Prediction

#### Random Forest

The provided code performs several tasks related to multi-output regression and machine learning model evaluation using the scikit-learn library and matplotlib for visualization. Here's a comprehensive summary of its main tasks and outputs:

### Data Loading and Preprocessing:

1. **Loading the Dataset:**
   - The code loads a dataset from a CSV file located at `D:\MTP\Mid Term\MId Term 256 models.csv` into a pandas DataFrame.

2. **Data Cleaning:**
   - Specific columns ("Sl No", "area", "Model horizontal Class", "Model Vertical class", "Model Box Class") are removed as they are unnecessary for the modeling process.

3. **Feature and Target Selection:**
   - Features and target variables are selected from the cleaned DataFrame.

### Model Training and Evaluation:

1. **Data Splitting:**
   - The dataset is split into training and testing sets using an 80-20 split ratio.

2. **Model Initialization:**
   - A RandomForestRegressor is initialized and used as the base estimator for a MultiOutputRegressor to handle multi-output regression.

3. **Hyperparameter Tuning:**
   - GridSearchCV is employed to perform hyperparameter tuning using a predefined parameter grid, aiming to find the best model configuration.

4. **Model Training and Predictions:**
   - The best model obtained from grid search is used to make predictions on the testing set.

5. **Evaluation Metrics Calculation:**
   - Several evaluation metrics are calculated for the testing set predictions, including:
     - Mean Squared Error (MSE)
     - Mean Absolute Error (MAE)
     - R-squared (R2)
   - These metrics provide insights into the model's performance and prediction accuracy.

### Management of Output Values:

1. **Storing Actual and Predicted Values:**
   - Actual and predicted values for each target variable (x1, y1, x2, y2) are stored in a DataFrame named `output_values`.
   - This DataFrame includes columns for both training and testing datasets' actual and predicted values.

2. **Evaluation Metrics for Individual Target Variables:**
   - Evaluation metrics are calculated separately for each target variable using the actual and predicted values stored in `output_values`.

### Output Management:

1. **Saving to CSV:**
   - The combined data of actual and predicted values is saved to a CSV file at a specified file path.
   - The saved data is then read back into a DataFrame for further analysis or verification.

### Visualization:

1. **Line Plots:**
   - Line plots are created to compare actual and predicted values for each target variable (x1, y1, x2, y2).

2. **Scatter Plots:**
   - Scatter plots are generated for a random subset of actual and predicted values of each target variable.

3. **Rectangular Plots:**
   - Rectangular plots are used to compare actual and predicted data points, providing a visual representation of the model's performance.

### Summary of Key Metrics:

- **Overall Metrics:**
  - MSE: 0.0195
  - MAE: 0.0975
  - R-squared: 75.59%

- **Individual Target Variables:**
  - **x1:**
    - MSE: 0.0070
    - MAE: 0.0504
    - R-squared: 90.79%
  - **y1:**
    - MSE: 0.0041
    - MAE: 0.0402
    - R-squared: 94.65%
  - **x2:**
    - MSE: 0.0071
    - MAE: 0.0507
    - R-squared: 90.73%
  - **y2:**
    - MSE: 0.0041
    - MAE: 0.0402
    - R-squared: 94.60%

#### CatBoostRegressor

The provided code involves various stages, including data loading, preprocessing, model training, evaluation, result visualization, and output management. Let's break down each aspect in detail:

## 1. Data Loading and Preprocessing

- **Libraries**: The code begins by importing necessary libraries such as pandas, numpy, sklearn, catboost, seaborn, and matplotlib.
- **Loading Data**: It loads a dataset from a CSV file using pandas' `read_csv()` function.
- **Dataset Information**: The loaded dataset contains 256 rows and 15 columns, representing different attributes.
- **Feature and Target Specification**: Features include six columns representing natural frequencies of modes, and targets include four columns representing coordinates (x1, y1, x2, y2).

## 2. Model Training and Evaluation

- **Model Training**: CatBoostRegressor models are trained individually for each target variable (x1, y1, x2, y2).
  - **Data Splitting**: For each target variable, the dataset is split into training and testing sets using `train_test_split()` from sklearn.
  - **Model Initialization and Training**: CatBoostRegressor is initialized with specific parameters and trained on the training set using the `fit()` method.
- **Model Evaluation**: The model is then evaluated on both the training and test sets using metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared. Metrics are printed for both the training and test sets.

## 3. Aggregation and Evaluation on Entire Dataset

- **Aggregation of Values**: Actual and predicted values for all target variables are aggregated for the entire dataset.
- **Evaluation Metrics**: MSE, MAE, and R-squared values are calculated for the entire dataset, combining both training and test sets.

## 4. Output Management

- **DataFrames for Outputs**: Actual and predicted values for the training and test sets are combined into separate DataFrames.
- **Concatenation and Saving**: These DataFrames are concatenated, and the 'Sl No' column from the original dataset is added. The combined DataFrame is saved to an Excel file using the `to_excel()` function.

## 5. Visualization

- **Scatter Plots**: Scatter plots are generated to visualize actual vs. predicted values for each target variable (x1, y1, x2, y2).
- **Combined Scatter Plot**: Additionally, a plot is created to visualize actual vs. predicted squares based on x and y coordinates.

## Summary of Results and Output Values


- **Metrics for Entire Dataset:**
  - **x1:**
    - MSE: 0.0021
    - MAE: 0.0162
    - R-squared: 97.28%
  - **y1:**
    - MSE: 0.0012
    - MAE: 0.0128
    - R-squared: 98.41%
  - **x2:**
    - MSE: 0.0021
    - MAE: 0.0162
    - R-squared: 97.28%
  - **y2:**
    - MSE: 0.0012
    - MAE: 0.0128
    - R-squared: 98.41%

## Conclusions and Scope for Future Work

### Conclusions

- Python scripting significantly enhanced data collection, preprocessing, and analysis efficiency.
- Four machine learning models provided valuable insights into damage detection.
- SVM outperformed KNN in classification tasks.
- CatBoost surpassed Random Forest in regression tasks with superior R-squared scores.

### Limitations

- Reliance on simulated data may not fully capture real-world complexities.
- Ensuring model generalizability requires diverse training datasets.
- Computational demands necessitate careful consideration of resource constraints and scalability.

### Scope for Future Work

- Expand beyond square plate shapes to explore applicability in diverse scenarios.
- Integrate advanced deep learning techniques like Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs).
- Subject models to real-world testing conditions to bolster credibility and robustness.
- Incorporate various sensors and data types for real-time structural health monitoring.
- Address uncertainties in data and models to enhance reliability.

## Contact Information

For inquiries, feedback, or support, please reach out to:

- **Email**: your-email@example.com
- **LinkedIn**: [Your Name](https://www.linkedin.com/in/yourprofile)

## Resources

- [Article on Composite Material Damage Detection](#)
- [Tutorial on Machine Learning for Structural Health Monitoring](#)
- [GitHub Repository of Related Tools](#)


## Contact Information

For inquiries, feedback, or support, please feel free to reach out to:
- **Email:** [your-email@example.com](mailto:your-email@example.com)
- **LinkedIn:** [Your Name](https://www.linkedin.com/in/your-name/)

---

## Resources

- [Article on Composite Material Damage Detection](link-to-article)
- [Tutorial on Machine Learning for Structural Health Monitoring](link-to-tutorial)
- [GitHub Repository of Related Tools](link-to-repository)
