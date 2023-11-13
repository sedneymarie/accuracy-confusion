#!pip install scikit-learn pandas

import pandas as pd

# Load your datasets
data1 = pd.read_csv('banana_data.csv')
data2 = pd.read_csv('cabbage_data.csv')
data3 = pd.read_csv('garlic_data.csv')
data4 = pd.read_csv('papaya_data.csv')

# Perform any necessary data preprocessing, such as combining datasets, feature scaling, etc.
# You may need to merge or transform the datasets to create a single dataset for training.

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Define your features and labels
X = data1[['Temperature', 'Humidity', 'MQ2GasSensor', 'MQ3GasSensor','MQ5GasSensor','MQ7GasSensor','MQ8GasSensor','MQ135GasSensor' ]]  # Adjust the feature columns as needed
y = data1['Spoilagelevel']  # Adjust the label column as needed

# Split your data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the KNN classifier
n_neighbors = 5  # Adjust the number of neighbors as needed
knn = KNeighborsClassifier(n_neighbors=n_neighbors)
knn.fit(X_train, y_train)

# Predict labels on the test set
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Create a confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='magma', xticklabels=['Not Spoiled', 'Average', 'Spoiled'], yticklabels=['Not Spoiled', 'Average', 'Spoiled'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

