import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset
train_df = pd.read_csv('data/train.csv')

# ============================================
# 1. GROUPBY ANALYSIS - Fare by Class
# ============================================

# Get fare statistics grouped by Pclass
fare_by_class = train_df.groupby('Pclass')['Fare'].agg([
    'count', 'mean', 'median', 'std', 'min', 'max'
]).round(2)

print("=" * 60)
print("FARE STATISTICS BY PASSENGER CLASS")
print("=" * 60)
print(fare_by_class)
print("\n")

# Get percentile thresholds for each class
percentiles = [25, 50, 75]
fare_percentiles = train_df.groupby('Pclass')['Fare'].quantile(
    [0.25, 0.50, 0.75]
).unstack()
fare_percentiles.columns = ['25th_percentile', '50th_percentile', '75th_percentile']

print("FARE PERCENTILES BY CLASS")
print(fare_percentiles)
print("\n")

# ============================================
# 2. VISUALIZATION - Fare Distribution
# ============================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Box plot
sns.boxplot(data=train_df, x='Pclass', y='Fare', ax=axes[0])
axes[0].set_title('Fare Distribution by Passenger Class', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Passenger Class')
axes[0].set_ylabel('Fare (£)')
axes[0].set_ylim(0, 300)  # Limit y-axis for better visualization

# Violin plot
sns.violinplot(data=train_df, x='Pclass', y='Fare', ax=axes[1])
axes[1].set_title('Fare Density by Passenger Class', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Passenger Class')
axes[1].set_ylabel('Fare (£)')
axes[1].set_ylim(0, 300)

plt.tight_layout()
plt.savefig('fare_distribution_by_class.png', dpi=300, bbox_inches='tight')
print("✓ Plot saved as 'fare_distribution_by_class.png'\n")
plt.show()

# ============================================
# 3. RULE-BASED CLASS CLASSIFIER
# ============================================

class PclassClassifier:
    """
    Rule-based classifier to assign Pclass based on 1912 fare amount
    Uses median values as thresholds from training data
    """
    
    def __init__(self, train_df):
        """Initialize with training data to calculate thresholds"""
        # Calculate class boundaries based on data
        self.class_1_min = train_df[train_df['Pclass'] == 1]['Fare'].quantile(0.25)
        self.class_2_median = train_df[train_df['Pclass'] == 2]['Fare'].median()
        self.class_3_max = train_df[train_df['Pclass'] == 3]['Fare'].quantile(0.75)
        
        # Set thresholds (using overlapping ranges, prioritize higher class for edge cases)
        self.threshold_1st = 30.0  # £30+ = 1st class
        self.threshold_2nd = 13.0  # £13-30 = 2nd class
        # Below £13 = 3rd class
        
        print(f"Classifier Initialized:")
        print(f"  1st Class threshold: ≥ £{self.threshold_1st}")
        print(f"  2nd Class threshold: £{self.threshold_2nd} - £{self.threshold_1st}")
        print(f"  3rd Class threshold: < £{self.threshold_2nd}")
        print()
    
    def predict_class(self, fare_1912_gbp):
        """
        Predict passenger class based on 1912 fare
        
        Args:
            fare_1912_gbp: Fare amount in 1912 British Pounds
            
        Returns:
            int: Passenger class (1, 2, or 3)
        """
        if fare_1912_gbp >= self.threshold_1st:
            return 1
        elif fare_1912_gbp >= self.threshold_2nd:
            return 2
        else:
            return 3
    
    def get_class_details(self, fare_1912_gbp):
        """
        Get detailed class assignment with context
        
        Returns:
            dict: Class assignment with details
        """
        pclass = self.predict_class(fare_1912_gbp)
        
        class_descriptions = {
            1: {
                'name': 'First Class',
                'description': 'Luxury accommodations with premium amenities',
                'typical_passengers': 'Wealthy elite, business magnates, aristocracy'
            },
            2: {
                'name': 'Second Class',
                'description': 'Comfortable middle-class accommodations',
                'typical_passengers': 'Professionals, merchants, skilled workers'
            },
            3: {
                'name': 'Third Class',
                'description': 'Basic but adequate steerage accommodations',
                'typical_passengers': 'Immigrants, laborers, working class families'
            }
        }
        
        return {
            'pclass': pclass,
            'fare_gbp': fare_1912_gbp,
            'class_name': class_descriptions[pclass]['name'],
            'description': class_descriptions[pclass]['description'],
            'typical_passengers': class_descriptions[pclass]['typical_passengers']
        }


# ============================================
# 4. TEST THE CLASSIFIER
# ============================================

# Initialize classifier
classifier = PclassClassifier(train_df)

# # Test cases
# test_fares = [5.0, 15.0, 50.0, 100.0]

# print("\n" + "=" * 60)
# print("TESTING CLASS CLASSIFIER")
# print("=" * 60)

# for fare in test_fares:
#     result = classifier.get_class_details(fare)
#     print(f"\nFare: £{fare:.2f}")
#     print(f"  → Class: {result['pclass']} ({result['class_name']})")
#     print(f"  → {result['description']}")
#     print(f"  → Typical passengers: {result['typical_passengers']}")

# print("\n" + "=" * 60)