#bagged trees
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import BaggingClassifier
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('train_final_labelled.csv')
test = pd.read_csv('test_final_labelled.csv')

# Split data into X and y
y = data['label']
X = data.drop('label', axis=1)

# Replace entries with question marks with the most common value in that column
for column in X.columns:
    if X[column].dtype == object:
        most_common_value = X[column].mode()[0]
    else:
        most_common_value = X[column].astype(str).mode()[0]
    X[column] = X[column].replace('?', most_common_value)

# Replace entries with question marks with the most common value in that column
for column in test.columns:
    if test[column].dtype == object:
        most_common_value = test[column].mode()[0]
    else:
        most_common_value = test[column].astype(str).mode()[0]
    test[column] = test[column].replace('?', most_common_value)

# Identify categorical columns
categorical_cols = test.select_dtypes(include=['object']).columns

# One-hot encode categorical columns
ct = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

X = ct.fit_transform(X).toarray()  # Convert to dense array
test = ct.transform(test).toarray()  # Convert to dense array

# Ensure X and test have the same number of columns
if X.shape[1] != test.shape[1]:
    max_columns = max(X.shape[1], test.shape[1])
    X = np.pad(X, ((0, 0), (0, max_columns - X.shape[1])), 'constant')
    test = np.pad(test, ((0, 0), (0, max_columns - test.shape[1])), 'constant')

print('X shape:', X.shape)
print('test shape:', test.shape)

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)

# # Train a decision tree
# dtree = DecisionTreeClassifier(random_state = 22)
# dtree.fit(X_train,y_train)

# # Predict the test and train data
# y_pred = dtree.predict(X_test)
# y_pred_train = dtree.predict(X_train)

# # Calculate the accuracy of the model
# accuracy = accuracy_score(y_test, y_pred)
# accuracy_train = accuracy_score(y_train, y_pred_train)
# print('Accuracy:', accuracy)
# print('Accuracy Train:', accuracy_train)

# estimator_range = [10,30,50]

# models = []
# scores = []

# for n_estimators in estimator_range:

#     # Create bagging classifier
#     clf = BaggingClassifier(n_estimators = n_estimators, random_state = 22)

#     # Fit the model
#     clf.fit(X, y)

#     # Append the model and score to their respective list
#     models.append(clf)
#     scores.append(accuracy_score(y_true = y_test, y_pred = clf.predict(X_test)))

# # Generate the plot of scores against number of estimators
# plt.figure(figsize=(9,6))
# plt.plot(estimator_range, scores)

# # Adjust labels and font (to make visable)
# plt.xlabel("n_estimators", fontsize = 18)
# plt.ylabel("score", fontsize = 18)
# plt.tick_params(labelsize = 16)

# # Visualize plot
# plt.show()

# Create bagging classifier
clf = BaggingClassifier(n_estimators = 50, random_state = 22)

# Fit the model
clf.fit(X, y)

# Predict the test and train data
y_pred = clf.predict(test)
y_pred_train = clf.predict(X)

# Calculate the accuracy of the model
accuracy = accuracy_score(y, y_pred_train)
print('Accuracy:', accuracy)

# Save the predictions to a CSV file
output = pd.DataFrame({'ID': np.arange(1, len(y_pred) + 1), 'Prediction': y_pred})
output.to_csv('bagged_out_1.csv', index=False)

# Save the predictions to a CSV file
output = pd.DataFrame({'ID': np.arange(1, len(y_pred) + 1), 'Prediction': y_pred})
output.to_csv('random_forest_out_1.csv', index=False)