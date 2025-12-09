import pandas as pd
import pickle

# Test 1: Load data
print("Test 1: Loading data...")
df = pd.read_csv('data/train.csv')
print(f"✓ Loaded {len(df)} passengers")

# Test 2: Load model
print("\nTest 2: Loading model...")
with open('models/titanic_model.pkl', 'rb') as f:
    model_data = pickle.load(f)
print("✓ Model loaded successfully")

# Test 3: Test imports
print("\nTest 3: Testing imports...")
from utils.currency_converter import TitanicCurrencyConverter
from utils.class_classifier import PclassClassifier
from utils.passenger_finder import NearestPassengerFinder
from utils.fare_calculator import ModernFareCalculator
print("✓ All imports working!")

print("\n🎉 Setup complete! Ready to build the app!")