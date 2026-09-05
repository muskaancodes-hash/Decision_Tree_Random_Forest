import pandas as pd

# Load dataset
df = pd.read_csv("heart.csv.xls")

# Display first 5 rows
print("First 5 rows:")
print(df.head())

# Dataset shape
print("\nDataset shape:")
print(df.shape)

# Column names
print("\nColumn names:")
print(df.columns)

# Dataset information
print("\nDataset information:")
print(df.info())

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Target distribution
print("\nTarget distribution:")
print(df["target"].value_counts())
# Part 2: Decision Tree Classifier

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Separate features and target
X = df.drop("target", axis=1)
y = df["target"]

# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)

# Train the model
dt_model.fit(X_train, y_train)

# Make predictions
y_pred = dt_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nDecision Tree Accuracy:", accuracy)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# Part 3: Analyze Overfitting and Control Tree Depth

from sklearn.metrics import accuracy_score

# Train Decision Trees with different maximum depths
depths = [1, 2, 3, 4, 5, 6, 8, 10]

print("\nTree Depth Analysis:")

for depth in depths:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)

    print(
        f"Depth {depth}: "
        f"Training Accuracy = {train_accuracy:.4f}, "
        f"Testing Accuracy = {test_accuracy:.4f}"
    )
    # Part 4: Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier

# Create Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
rf_pred = rf_model.predict(X_test)

# Calculate Random Forest accuracy
rf_accuracy = accuracy_score(y_test, rf_pred)

print("\nRandom Forest Accuracy:", rf_accuracy)

# Compare both models
print("\nModel Accuracy Comparison:")
print("Decision Tree Accuracy:", accuracy)
print("Random Forest Accuracy:", rf_accuracy)
# Part 5: Feature Importance

import matplotlib.pyplot as plt

# Get feature importance from Random Forest
feature_importance = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

# Display feature importance
print("\nFeature Importance:")
print(feature_importance)

# Plot feature importance
plt.figure(figsize=(10, 6))
feature_importance.plot(kind="bar")
plt.title("Feature Importance - Random Forest")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# Part 6: Cross-Validation

from sklearn.model_selection import cross_val_score

# Cross-validation for Decision Tree
dt_scores = cross_val_score(
    dt_model, X, y, cv=5, scoring="accuracy"
)

# Cross-validation for Random Forest
rf_scores = cross_val_score(
    rf_model, X, y, cv=5, scoring="accuracy"
)

# Display results
print("\nDecision Tree Cross-Validation Scores:")
print(dt_scores)

print("\nDecision Tree Mean CV Accuracy:")
print(dt_scores.mean())

print("\nRandom Forest Cross-Validation Scores:")
print(rf_scores)

print("\nRandom Forest Mean CV Accuracy:")
print(rf_scores.mean())
# Part 7: Decision Tree Visualization

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Create a limited-depth tree for visualization
tree_visual = DecisionTreeClassifier(max_depth=3, random_state=42)

# Train the tree
tree_visual.fit(X_train, y_train)

# Plot the Decision Tree
plt.figure(figsize=(20, 10))

plot_tree(
    tree_visual,
    feature_names=X.columns,
    class_names=["No Disease", "Disease"],
    filled=True,
    rounded=True
)

plt.title("Decision Tree Visualization")
plt.show()