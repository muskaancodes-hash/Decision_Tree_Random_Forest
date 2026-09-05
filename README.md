# Decision Tree and Random Forest

## Project Description

This project implements Decision Tree and Random Forest classification models using the Heart Disease dataset. The project focuses on model training, overfitting control, feature importance, visualization, and cross-validation.

## Objectives

* Train a Decision Tree Classifier.
* Visualize the Decision Tree.
* Analyze and control overfitting using tree depth.
* Train a Random Forest Classifier.
* Compare Decision Tree and Random Forest accuracy.
* Interpret feature importances.
* Evaluate models using cross-validation.

## Tools and Technologies

* Python
* Pandas
* Scikit-learn
* Matplotlib
* Graphviz

## Dataset

The project uses the Heart Disease dataset.

Dataset file:

`heart.csv.xls`

---

## Part 1: Load and Explore Dataset

The Heart Disease dataset is loaded using Pandas.

### Steps Performed

* Loaded the dataset.
* Displayed the first five rows.
* Checked the dataset shape.
* Displayed column names.
* Viewed dataset information and data types.
* Checked for missing values.
* Analyzed the target variable distribution.

### Purpose

This step helps understand the structure and quality of the dataset before applying machine learning models.

---

## Part 2: Decision Tree Classifier

A Decision Tree Classifier is trained to classify the target variable.

### Steps Performed

* Separated features and target variable.
* Split the dataset into training and testing sets.
* Trained a Decision Tree Classifier.
* Generated predictions.
* Calculated accuracy.
* Generated a classification report.

### Result

The Decision Tree achieved approximately 99% accuracy on the test dataset.

---

## Part 3: Overfitting and Tree Depth Control

Different tree depth values are tested to analyze overfitting.

### Steps Performed

* Tested different `max_depth` values.
* Trained a Decision Tree for each depth.
* Calculated training accuracy.
* Calculated testing accuracy.
* Compared training and testing performance.

### Purpose

Controlling tree depth helps reduce model complexity and prevent overfitting.

---

## Part 4: Random Forest Classifier

A Random Forest Classifier is trained and compared with the Decision Tree.

### Steps Performed

* Created a Random Forest Classifier.
* Used 100 decision trees.
* Trained the Random Forest model.
* Generated predictions.
* Calculated Random Forest accuracy.
* Compared Random Forest and Decision Tree accuracy.

### Purpose

Random Forest combines multiple decision trees to improve generalization and reduce overfitting.

---

## Part 5: Feature Importance

Feature importance is analyzed using the Random Forest model.

### Steps Performed

* Extracted feature importance values.
* Associated importance values with feature names.
* Sorted features by importance.
* Displayed feature importance values.
* Created a feature importance bar chart.

### Purpose

Feature importance helps identify which features contribute most to the model's predictions.

---

## Part 6: Cross-Validation

Five-fold cross-validation is used to evaluate both models.

### Steps Performed

* Applied 5-fold cross-validation to the Decision Tree.
* Applied 5-fold cross-validation to the Random Forest.
* Calculated accuracy for each fold.
* Calculated mean cross-validation accuracy.
* Compared the models.

### Purpose

Cross-validation provides a more reliable estimate of model performance by evaluating the model on different portions of the dataset.

---

## Part 7: Decision Tree Visualization

The Decision Tree is visualized using Scikit-learn's `plot_tree()` function.

### Steps Performed

* Created a Decision Tree with `max_depth=3`.
* Trained the tree.
* Used `plot_tree()` for visualization.
* Displayed feature names and class names.
* Generated a graphical representation of the decision-making process.

### Purpose

Tree visualization helps understand how the Decision Tree makes classification decisions using different features and conditions.

---

## Conclusion

This project demonstrates the use of Decision Tree and Random Forest algorithms for heart disease classification. It also covers overfitting control, model comparison, feature importance, tree visualization, and cross-validation.

The project provides practical understanding of tree-based machine learning models and their evaluation.
