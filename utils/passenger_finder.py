import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

class NearestPassengerFinder:
    """
    Find the closest matching passenger from Titanic dataset
    Based on: Age, Sex, Pclass
    NO SibSp or Parch (as per requirements)
    """
    
    def __init__(self, train_df):
        """
        Initialize with training data
        
        Args:
            train_df: Titanic training dataset
        """
        self.train_df = train_df.copy()
        
        # Prepare the data
        self._prepare_data()
        
        print(f"✓ Loaded {len(self.clean_df)} passengers for matching")
        print(f"  Features used: Age, Sex, Pclass")
        print()
    
    def _prepare_data(self):
        """Prepare and clean data for similarity matching"""
        
        # Select only passengers with complete Age data
        self.clean_df = self.train_df[self.train_df['Age'].notna()].copy()
        
        # Encode Sex: male=1, female=0
        self.clean_df['Sex_encoded'] = (self.clean_df['Sex'] == 'male').astype(int)
        
        # Create feature matrix
        self.feature_cols = ['Age', 'Sex_encoded', 'Pclass']
        self.features = self.clean_df[self.feature_cols].values
        
        # Normalize features for fair distance calculation
        self.scaler = StandardScaler()
        self.features_scaled = self.scaler.fit_transform(self.features)
        
    def find_nearest(self, age, sex, pclass, top_n=1):
        """
        Find the nearest passenger(s) based on user inputs
        
        Args:
            age: User's age (int or float)
            sex: User's sex ('male' or 'female')
            pclass: Predicted passenger class (1, 2, or 3)
            top_n: Number of nearest passengers to return (default: 1)
            
        Returns:
            dict or list of dicts with nearest passenger details
        """
        
        # Encode user input
        sex_encoded = 1 if sex.lower() == 'male' else 0
        
        # Create user feature vector
        user_features = np.array([[age, sex_encoded, pclass]])
        
        # Scale using the same scaler
        user_features_scaled = self.scaler.transform(user_features)
        
        # Calculate distances to all passengers
        distances = euclidean_distances(user_features_scaled, self.features_scaled)[0]
        
        # Find nearest passenger(s)
        nearest_indices = np.argsort(distances)[:top_n]
        
        results = []
        for idx in nearest_indices:
            original_idx = self.clean_df.index[idx]
            passenger = self.train_df.loc[original_idx]
            
            result = {
                'PassengerId': int(passenger['PassengerId']),
                'Name': passenger['Name'],
                'Age': int(passenger['Age']),
                'Sex': passenger['Sex'],
                'Pclass': int(passenger['Pclass']),
                'Fare': round(float(passenger['Fare']), 2),
                'Survived': int(passenger['Survived']),
                'Survived_text': 'Survived ✓' if passenger['Survived'] == 1 else 'Did not survive ✗',
                'distance': round(float(distances[idx]), 4),
                'Embarked': passenger.get('Embarked', 'Unknown'),
                'SibSp': int(passenger['SibSp']) if pd.notna(passenger['SibSp']) else 0,
                'Parch': int(passenger['Parch']) if pd.notna(passenger['Parch']) else 0
            }
            results.append(result)
        
        return results[0] if top_n == 1 else results
    
    def get_match_summary(self, age, sex, pclass):
        """
        Get detailed summary of the nearest passenger match
        
        Returns:
            dict: Complete match details with prediction
        """
        match = self.find_nearest(age, sex, pclass)
        
        # Calculate survival probability based on similar passengers
        similar_passengers = self.find_nearest(age, sex, pclass, top_n=10)
        survival_rate = sum(p['Survived'] for p in similar_passengers) / len(similar_passengers)
        
        summary = {
            'user_input': {
                'age': age,
                'sex': sex,
                'pclass': pclass
            },
            'nearest_match': match,
            'survival_probability': round(survival_rate * 100, 1),
            'confidence': self._get_confidence_level(match['distance'])
        }
        
        return summary
    
    def _get_confidence_level(self, distance):
        """
        Determine match confidence based on distance
        
        Args:
            distance: Euclidean distance to nearest match
            
        Returns:
            str: Confidence level (High, Medium, Low)
        """
        if distance < 0.5:
            return 'High'
        elif distance < 1.5:
            return 'Medium'
        else:
            return 'Low'
    
    def print_match_details(self, age, sex, pclass):
        """Print formatted match details"""
        
        summary = self.get_match_summary(age, sex, pclass)
        match = summary['nearest_match']
        
        print("\n" + "=" * 70)
        print("YOUR NEAREST TITANIC PASSENGER MATCH")
        print("=" * 70)
        
        print(f"\n📝 Your Profile:")
        print(f"   Age: {age} | Sex: {sex.capitalize()} | Class: {pclass}")
        
        print(f"\n👤 Closest Match: {match['Name']}")
        print(f"   Passenger ID: {match['PassengerId']}")
        print(f"   Age: {match['Age']} | Sex: {match['Sex'].capitalize()} | Class: {match['Pclass']}")
        print(f"   Fare Paid: £{match['Fare']}")
        print(f"   Embarked: {match['Embarked']}")
        
        if match['SibSp'] > 0 or match['Parch'] > 0:
            print(f"   Traveling with: {match['SibSp']} siblings/spouse, {match['Parch']} parents/children")
        else:
            print(f"   Traveling: Alone")
        
        print(f"\n🎯 Match Quality: {summary['confidence']} confidence (distance: {match['distance']})")
        
        print(f"\n⚠️  Historical Outcome: {match['Survived_text']}")
        print(f"   Based on 10 similar passengers: {summary['survival_probability']}% survival rate")
        
        print("=" * 70 + "\n")
        
        return summary


# ============================================
# EXAMPLE USAGE
# ============================================

if __name__ == "__main__":
    # Load data
    train_df = pd.read_csv('train.csv')
    
    # Initialize finder
    finder = NearestPassengerFinder(train_df)
    
    # # Test cases
    # print("\n🔍 TEST CASE 1: Young male, 3rd class")
    # finder.print_match_details(age=22, sex='male', pclass=3)
    
    # print("\n🔍 TEST CASE 2: Middle-aged female, 1st class")
    # finder.print_match_details(age=35, sex='female', pclass=1)
    
    # print("\n🔍 TEST CASE 3: Young female, 2nd class")
    # finder.print_match_details(age=25, sex='female', pclass=2)
    
    # # Get raw match data
    # print("\n📊 Getting raw match data...")
    # summary = finder.get_match_summary(age=28, sex='male', pclass=2)
    # print(f"Match: {summary['nearest_match']['Name']}")
    # print(f"Survival probability: {summary['survival_probability']}%")