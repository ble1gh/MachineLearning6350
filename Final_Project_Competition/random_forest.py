import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor


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

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)

# state = [5, 44, 111, 500, 1000, 2000, 5000, 10000]

# models = []
# scores = []

# for i in state:
#     # Train a random forest model
#     model = RandomForestRegressor(n_estimators=i, random_state=22)
#     model.fit(X_train, y_train)

#     # Make predictions
#     y_pred = model.predict(X_test)
#     y_pred_train = model.predict(X_train)

#     # Append the model and score to their respective list
#     models.append(model)
#     y_pred = model.predict(X_test)
#     scores.append(accuracy_score(y_true = y_test, y_pred = y_pred.round()))

# # Generate the plot of scores against number of estimators
# plt.figure(figsize=(9,6))
# plt.plot(state, scores)

# # Adjust labels and font (to make visable)
# plt.xlabel("random state", fontsize = 18)
# plt.ylabel("score", fontsize = 18)
# plt.tick_params(labelsize = 16)

# # Visualize plot
# plt.show()

# Train a random forest model
model = RandomForestRegressor(n_estimators=5000, random_state=22)
model.fit(X, y)

# Make predictions
y_pred = model.predict(test)
y_pred_train = model.predict(X)

# Calculate the accuracy of the model
train_accuracy = accuracy_score(y, y_pred_train.round())
print('Train Accuracy:', train_accuracy)

# Save the predictions to a CSV file
output = pd.DataFrame({'ID': np.arange(1, len(y_pred) + 1), 'Prediction': y_pred})
output.to_csv('random_forest_out_2.csv', index=False)
