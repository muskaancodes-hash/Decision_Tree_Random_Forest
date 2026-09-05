# Decision_Tree_Random_Forest
### Description  This part loads the Heart Disease dataset using Pandas and performs basic data exploration. It displays the first five rows, dataset shape, column names, dataset information, missing values, and target variable distribution.
# Heart Disease Dataset Exploration

## Overview

This part focuses on loading and exploring the Heart Disease dataset using Python and Pandas.

## Tasks Performed

* Loaded the `heart.csv.xls` dataset.
* Displayed the first five rows.
* Checked the dataset shape.
* Displayed column names.
* Viewed dataset information and data types.
* Checked for missing values.
* Analyzed the target variable distribution.

## Tools Used

* Python
* Pandas

## Purpose

The purpose of this step is to understand the dataset before applying Decision Tree and Random Forest machine learning models.
## Part 2: Decision Tree Classifier

In this step, a Decision Tree Classifier is trained using the Heart Disease dataset.

### Steps Performed

* Separated features and target variable.
* Split the dataset into training and testing sets.
* Trained a Decision Tree Classifier.
* Generated predictions on the test data.
* Calculated model accuracy.
* Generated a classification report.

### Result

The Decision Tree achieved approximately **99% accuracy** on the test dataset.
## Part 3: Overfitting and Tree Depth Control

In this step, the effect of Decision Tree depth on model performance is analyzed.

### Steps Performed

* Tested different `max_depth` values.
* Trained a Decision Tree for each depth.
* Calculated training accuracy.
* Calculated testing accuracy.
* Compared training and testing performance.
* Used the results to identify possible overfitting.

### Purpose

A Decision Tree can become too complex and memorize the training data, causing overfitting. Controlling the `max_depth` helps create a simpler model that can generalize better to unseen data.

### Result

The training and testing accuracy values for different tree depths are compared to understand the relationship between tree complexity and model performance.
## Part 4: Random Forest Classifier

In this step, a Random Forest Classifier is trained and its performance is compared with the Decision Tree model.

### Steps Performed

* Created a Random Forest Classifier.
* Used 100 decision trees with `n_estimators=100`.
* Trained the Random Forest using the training dataset.
* Generated predictions on the test dataset.
* Calculated Random Forest accuracy.
* Compared Random Forest accuracy with Decision Tree accuracy.

### Purpose

Random Forest is an ensemble learning method that combines multiple decision trees. It generally provides better generalization and helps reduce overfitting compared with a single Decision Tree.

### Result

The accuracy of the Decision Tree and Random Forest models is compared to determine which model performs better on the Heart Disease dataset.
## Part 5: Feature Importance

In this step, the importance of individual features in the Random Forest model is analyzed.

### Steps Performed

* Extracted feature importance values from the Random Forest model.
* Associated importance values with the corresponding feature names.
* Sorted the features from most important to least important.
* Displayed the feature importance values.
* Created a bar chart to visualize feature importance.

### Purpose

Feature importance helps understand which input features have the greatest influence on the Random Forest model's predictions.

### Result

A feature importance chart is generated, showing the relative contribution of each feature to the Random Forest classification model.


### Purpose

This step demonstrates how a Decision Tree can be used for heart disease classification and how its performance can be evaluated using standard classification metrics.
