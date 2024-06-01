# Damage Detection in Composite Plates

Welcome to the repository documenting a pioneering research project focused on damage detection in Glass Fiber Reinforced Polymer (GFRP) composite plates, leveraging cutting-edge machine learning methodologies. This project, undertaken as part of the Master of Technology program in Civil Engineering with a specialization in Structural Engineering at the esteemed Indian Institute of Technology Kharagpur, represents a significant advancement in structural health monitoring techniques.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Components](#key-components)
- [Project Workflow](#project-workflow)
- [Results and Discussions](#results-and-discussions)
  - [Data Collection and Scripting Efficiency](#data-collection-and-scripting-efficiency)
  - [Performance of Classification Algorithms](#performance-of-classification-algorithms)
  - [Performance of Regression Algorithms](#performance-of-regression-algorithms)
- [Conclusions and Scope for Future Work](#conclusions-and-scope-for-future-work)
  - [Conclusions](#conclusions)
  - [Limitations](#limitations)
  - [Scope for Future Work](#scope-for-future-work)
- [Contact Information](#contact-information)
- [Resources](#resources)

---

## Project Overview

**Objective:** Revolutionize traditional damage detection methods by integrating state-of-the-art machine learning algorithms to automate and optimize the efficiency of damage detection processes.

---

## Key Components

- **Data Collection and Preprocessing:**
  - Utilized experimental modal testing and finite element simulations to collect extensive data on the structural response of GFRP composite plates.
  - Employed Python scripting for preprocessing raw data and extracting pertinent features for machine learning models.

- **Classification Algorithms for Damage Detection:**
  - Implemented algorithms like K-Nearest Neighbors (KNN) and Support Vector Machine (SVM) to classify damage classes within predefined regions of the composite plates.
  - Evaluated these algorithms based on accuracy and other relevant performance metrics.

- **Regression Algorithms for Damage Severity Prediction:**
  - Used regression algorithms such as Random Forest and CatBoost to predict damage severity and coordinates within the composite plates.
  - These models provide insights into the extent and precise location of damage, facilitating targeted maintenance and repair strategies.

---

## Project Workflow

- **Data Collection:** Conducted experimental modal testing and finite element simulations to gather comprehensive data on natural frequencies and structural responses.
- **Data Preprocessing:** Preprocessed raw data from modal testing and simulations to extract relevant features and target variables for machine learning models.
- **Model Training:** Trained classification and regression algorithms using the preprocessed data to develop predictive models for damage detection and severity assessment.
- **Model Evaluation:** Evaluated model performance using metrics such as accuracy, precision, recall, mean squared error (MSE), and R-squared score.
- **Result Analysis:** Analyzed results from different algorithms to identify strengths, weaknesses, and areas for further refinement in damage detection methodologies.

---

## Results and Discussions

### Data Collection and Scripting Efficiency

- Python scripting was crucial for:
  - Automating data collection and preparation in Abaqus.
  - Extracting data like natural frequencies and class labels directly from simulations, reducing manual errors.

### Performance of Classification Algorithms

- **K-Nearest Neighbors (KNN):**
  - Achieved high accuracy across all damage classes.
  - Accuracy: 79.90% (horizontal), 81.37% (vertical), 79.41% (box).

- **Support Vector Machine (SVM):**
  - Demonstrated superior precision and recall metrics.
  - Indicated proficiency in correctly identifying damage classes.

### Performance of Regression Algorithms

- **Random Forest:**
  - Mean Squared Error (MSE): 0.0195
  - Mean Absolute Error (MAE): 0.0975
  - R-squared: 75.59%

- **CatBoost:**
  - MSE: 0.015
  - MAE: 0.08
  - Average R-squared: 97.28%
  - Excelled in handling categorical features and capturing complex data relationships.

---

## Conclusions and Scope for Future Work

### Conclusions

- Python scripting significantly enhanced the efficiency and reliability of data collection, preprocessing, and analysis.
- Four machine learning models (two classification and two regression) provided valuable insights into damage detection potential.
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

---

## Contact Information

For inquiries, feedback, or support, please feel free to reach out to:
- **Email:** [your-email@example.com](mailto:your-email@example.com)
- **LinkedIn:** [Your Name](https://www.linkedin.com/in/your-name/)

---

## Resources

- [Article on Composite Material Damage Detection](link-to-article)
- [Tutorial on Machine Learning for Structural Health Monitoring](link-to-tutorial)
- [GitHub Repository of Related Tools](link-to-repository)
