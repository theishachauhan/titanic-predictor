# import os
# import pickle
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt


# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier

# # Set a random seed
# import random
# random.seed(42)
# data = pd.read_csv('data/train.csv')

# # Print the first few entries of the Titanic data
# data.head()

# outcomes = data['Survived']
# features = data.drop('Survived', axis=1)

# features_encoded = pd.get_dummies(features)

# features_encoded = features_encoded.fillna(0.0)
# features_encoded.head()

# X_train, X_test, y_train, y_test = train_test_split(features_encoded, outcomes, test_size=0.2, random_state=42)

# model = DecisionTreeClassifier()
# model.fit(X_train, y_train)

# # Making Predictions
# y_train_pred = model.predict(X_train)
# y_test_pred = model.predict(X_test)

# # Calculate the accuracy
# from sklearn.metrics import accuracy_score
# train_accuracy = accuracy_score(y_train, y_train_pred)
# test_accuracy = accuracy_score(y_test, y_test_pred)
# print('The training accuracy is', train_accuracy)
# print('The test accuracy is', test_accuracy)

# from sklearn.model_selection import GridSearchCV

# model = DecisionTreeClassifier(random_state=42)

# params = {
#     'max_depth': range(2,11),
#     'min_samples_leaf': range(2,11),
#     'min_samples_split': range(2,11)
# }

# grid_search = GridSearchCV(model, params)
# grid_search.fit(X_train, y_train)

# print(grid_search.best_params_)

# # Make predictions
# # predict() on GridSearchCV picks the best model
# y_train_pred = grid_search.predict(X_train)
# y_test_pred = grid_search.predict(X_test)

# # Calculate the accuracy
# from sklearn.metrics import accuracy_score
# train_accuracy = accuracy_score(y_train, y_train_pred)
# test_accuracy = accuracy_score(y_test, y_test_pred)
# print('The training accuracy is', train_accuracy)
# print('The test accuracy is', test_accuracy)
# with open("best_model.pkl", "wb") as f:
#     pickle.dump(grid_search.best_estimator_, f)

# print("Model saved as best_model.pkl")

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# Set a random seed
import random
random.seed(42)

print("=" * 60)
print("TRAINING TITANIC SURVIVAL MODEL")
print("=" * 60)

# Load data
print("\n1. Loading Titanic dataset...")
data = pd.read_csv('data/train.csv')
print(f"   ✓ Loaded {len(data)} passengers")

# Print the first few entries of the Titanic data
print("\n2. Data preview:")
print(data.head())

# Separate outcomes and features
outcomes = data['Survived']
features = data.drop('Survived', axis=1)

# Encode categorical features (one-hot encoding)
print("\n3. Encoding features with one-hot encoding...")
features_encoded = pd.get_dummies(features)
features_encoded = features_encoded.fillna(0.0)
print(f"   ✓ Total features after encoding: {len(features_encoded.columns)}")

# Split the data
print("\n4. Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    features_encoded, outcomes, test_size=0.2, random_state=42
)
print(f"   ✓ Training set: {len(X_train)} passengers")
print(f"   ✓ Testing set: {len(X_test)} passengers")

# Initial model training
print("\n5. Training initial Decision Tree model...")
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Making Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate the accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"   Initial training accuracy: {train_accuracy:.2%}")
print(f"   Initial test accuracy: {test_accuracy:.2%}")

# Grid Search for hyperparameter tuning
print("\n6. Running Grid Search for best hyperparameters...")
print("   (This may take a minute...)")

model = DecisionTreeClassifier(random_state=42)

params = {
    'max_depth': range(2, 11),
    'min_samples_leaf': range(2, 11),
    'min_samples_split': range(2, 11)
}

grid_search = GridSearchCV(model, params, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"   ✓ Best parameters found: {grid_search.best_params_}")

# Make predictions with best model
print("\n7. Evaluating best model...")
y_train_pred = grid_search.predict(X_train)
y_test_pred = grid_search.predict(X_test)

# Calculate the accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"   ✓ Optimized training accuracy: {train_accuracy:.2%}")
print(f"   ✓ Optimized test accuracy: {test_accuracy:.2%}")

# Save model as dictionary with metadata (for app compatibility)
print("\n8. Saving model...")

# Store the feature columns for reference
feature_columns = list(features_encoded.columns)

model_data = {
    'model': grid_search.best_estimator_,
    'feature_columns': feature_columns,
    'best_params': grid_search.best_params_,
    'test_accuracy': test_accuracy,
    'version': '1.0'
}

# Save to models directory
os.makedirs('models', exist_ok=True)

with open("models/titanic_model.pkl", "wb") as f:
    pickle.dump(model_data, f)

print("   ✓ Model saved as 'models/titanic_model.pkl'")
print(f"   ✓ Final test accuracy: {test_accuracy:.2%}")

# Also save the old format for backwards compatibility
with open("models/best_model.pkl", "wb") as f:
    pickle.dump(grid_search.best_estimator_, f)
print("   ✓ Also saved as 'models/best_model.pkl' (backup)")

print("\n" + "=" * 60)
print("✓ MODEL TRAINING COMPLETE!")
print("=" * 60)
print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Test accuracy: {test_accuracy:.2%}")
print(f"Total features: {len(feature_columns)}")
print("\nYou can now run: streamlit run app.py")
print("=" * 60)