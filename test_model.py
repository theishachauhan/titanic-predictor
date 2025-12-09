# test_model.py
import pickle

with open('models/titanic_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

print("Model type:", type(model_data))
print("\nKeys in model_data:", model_data.keys() if isinstance(model_data, dict) else "Not a dict!")

if isinstance(model_data, dict):
    print("\nFeature columns (first 10):", model_data['feature_columns'][:10])
    print("Total features:", len(model_data['feature_columns']))