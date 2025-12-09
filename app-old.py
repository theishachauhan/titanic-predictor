import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
from utils.currency_converter import TitanicCurrencyConverter
from utils.class_classifier import PclassClassifier
from utils.passenger_finder import NearestPassengerFinder
from utils.fare_calculator import ModernFareCalculator

# Page config
st.set_page_config(
    page_title="Would You Survive the Titanic?",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .title-text {
        font-size: 3.5rem;
        font-weight: bold;
        text-align: center;
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin-bottom: 1rem;
    }
    .subtitle-text {
        font-size: 1.3rem;
        text-align: center;
        color: #f0f0f0;
        margin-bottom: 2rem;
    }
    .result-card {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin: 1rem 0;
    }
    .survived-animation {
        text-align: center;
        padding: 3rem;
        background: linear-gradient(to bottom, #4facfe 0%, #00f2fe 100%);
        border-radius: 15px;
        animation: float 3s ease-in-out infinite;
    }
    .died-animation {
        text-align: center;
        padding: 3rem;
        background: linear-gradient(to bottom, #0f2027 0%, #203a43 50%, #2c5364 100%);
        border-radius: 15px;
        animation: sink 3s ease-in-out infinite;
    }
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-20px); }
    }
    @keyframes sink {
        0%, 100% { transform: translateY(0px); opacity: 1; }
        50% { transform: translateY(10px); opacity: 0.7; }
    }
    .character {
        font-size: 5rem;
        animation: wave 2s ease-in-out infinite;
    }
    @keyframes wave {
        0%, 100% { transform: rotate(0deg); }
        25% { transform: rotate(-10deg); }
        75% { transform: rotate(10deg); }
    }
    .bubble {
        display: inline-block;
        font-size: 2rem;
        animation: bubble-rise 3s ease-in infinite;
        opacity: 0.6;
    }
    @keyframes bubble-rise {
        0% { transform: translateY(0px); opacity: 0; }
        50% { opacity: 0.6; }
        100% { transform: translateY(-100px); opacity: 0; }
    }
    .stat-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stat-number { font-size: 2.5rem; font-weight: bold; }
    .stat-label { font-size: 1rem; opacity: 0.9; }
</style>
""", unsafe_allow_html=True)

# Load model and data
@st.cache_resource
def load_model():
    with open('models/titanic_model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    return pd.read_csv('data/train.csv')

# Initialize
try:
    model_data = load_model()
    train_df = load_data()
    currency_converter = TitanicCurrencyConverter()
    class_classifier = PclassClassifier(train_df)
    passenger_finder = NearestPassengerFinder(train_df)
    fare_calculator = ModernFareCalculator()
except Exception as e:
    st.error(f"Error loading model or data: {e}")
    st.stop()

# Header
st.markdown('<div class="title-text">🚢 Would YOU Survive the Titanic? 🌊</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Travel back to 1912 and discover your fate aboard the RMS Titanic</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("📝 Your Information")
st.sidebar.markdown("---")

age = st.sidebar.slider("Your Age", 1, 100, 25)
sex = st.sidebar.selectbox("Your Sex", ["Male", "Female"])
currency = st.sidebar.selectbox("Your Currency", ["USD", "GBP", "EUR", "INR"])
annual_earnings = st.sidebar.number_input(
    f"Annual Earnings ({currency})",
    min_value=1000,
    max_value=10000000,
    value=50000,
    step=1000
)

st.sidebar.markdown("---")
predict_button = st.sidebar.button("🎯 Predict My Fate!", use_container_width=True)

# Main content
if predict_button:
    with st.spinner("⏳ Calculating your fate..."):
        time.sleep(1)
        
        try:
            # Phase 1: Convert earnings to 1912 fare
            conversion_result = currency_converter.modern_to_1912_fare(annual_earnings, currency)
            fare_1912_gbp = conversion_result['fare_1912_gbp']
            
            # Phase 2: Determine passenger class
            class_result = class_classifier.get_class_details(fare_1912_gbp)
            pclass = class_result['pclass']
            
            # Phase 3: Find nearest passenger
            passenger_match = passenger_finder.get_match_summary(age, sex.lower(), pclass)
            nearest = passenger_match['nearest_match']
            survival_prob = passenger_match['survival_probability']
            
            # Phase 4: Calculate modern fare
            modern_fare = fare_calculator.fare_1912_to_modern(fare_1912_gbp, currency)
            
            # PREDICTION: Use nearest passenger's profile with user's overrides
            nearest_passenger_id = nearest['PassengerId']
            base_passenger = train_df[train_df['PassengerId'] == nearest_passenger_id].copy()
            
            # Override user-specific fields
            base_passenger.loc[:, 'Age'] = age
            base_passenger.loc[:, 'Sex'] = sex.lower()
            base_passenger.loc[:, 'Fare'] = fare_1912_gbp
            base_passenger.loc[:, 'Pclass'] = pclass
            
            # One-hot encode
            input_encoded = pd.get_dummies(base_passenger)
            input_encoded = input_encoded.fillna(0.0)
            
            # Get model
            if isinstance(model_data, dict):
                model = model_data['model']
                training_columns = model_data['feature_columns']
            else:
                st.error("Model format error. Please retrain the model.")
                st.stop()
            
            # Add missing columns
            for col in training_columns:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0
            
            # Reorder to match training
            input_encoded = input_encoded[training_columns]
            
            # Predict
            prediction = model.predict(input_encoded)[0]
            prediction_proba = model.predict_proba(input_encoded)[0]
            survival_model_prob = round(prediction_proba[1] * 100, 1)
            
            # Character selection
            if sex.lower() == 'female':
                if age < 18:
                    character = "👧"
                    character_name = "Young Kate"
                elif age < 40:
                    character = "👩"
                    character_name = "Rose DeWitt Bukater"
                else:
                    character = "👵"
                    character_name = "Older Rose"
            else:
                if age < 18:
                    character = "👦"
                    character_name = "Young Jack"
                elif age < 40:
                    character = "🧑"
                    character_name = "Jack Dawson"
                else:
                    character = "👴"
                    character_name = "Older Gentleman"
            
            # Display results
            st.markdown("---")
            
            # Animation
            if prediction == 1:
                st.markdown(f"""
                <div class="survived-animation">
                    <div class="character">{character}</div>
                    <h1 style="color: white;">🎉 YOU SURVIVED! 🎉</h1>
                    <h3 style="color: white;">You made it to New York!</h3>
                    <p style="color: white; font-size: 1.2rem;">
                        As {character_name}, you sailed safely across the Atlantic
                    </p>
                    <div style="margin-top: 2rem;">
                        <span style="font-size: 3rem;">🚢</span>
                        <span style="font-size: 2rem;">➡️</span>
                        <span style="font-size: 3rem;">🗽</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="died-animation">
                    <div class="character">{character}</div>
                    <h1 style="color: #ff6b6b;">💔 YOU DIDN'T SURVIVE 💔</h1>
                    <h3 style="color: #ffaaaa;">Lost in the Atlantic...</h3>
                    <p style="color: #cccccc; font-size: 1.2rem;">
                        As {character_name}, you went down with the ship
                    </p>
                    <div style="margin-top: 2rem;">
                        <span class="bubble">💧</span>
                        <span class="bubble" style="animation-delay: 0.5s;">💧</span>
                        <span class="bubble" style="animation-delay: 1s;">💧</span>
                        <span class="bubble" style="animation-delay: 1.5s;">💧</span>
                        <span class="bubble" style="animation-delay: 2s;">💧</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Stats
            st.markdown("### 📊 Your Titanic Profile")
            col1, col2, col3, col4 = st.columns(4)
            
            symbols = {'USD': '$', 'GBP': '£', 'EUR': '€', 'INR': '₹'}
            symbol = symbols.get(currency, '')
            
            with col1:
                st.markdown(f'<div class="stat-box"><div class="stat-number">{pclass}</div><div class="stat-label">Passenger Class</div></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="stat-box"><div class="stat-number">{survival_model_prob}%</div><div class="stat-label">Survival Probability</div></div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="stat-box"><div class="stat-number">{symbol}{modern_fare["modern_value_original_currency"]:,.0f}</div><div class="stat-label">Modern Ticket Price</div></div>', unsafe_allow_html=True)
            with col4:
                st.markdown(f'<div class="stat-box"><div class="stat-number">£{fare_1912_gbp:.0f}</div><div class="stat-label">1912 Fare</div></div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Details
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.markdown("### 🎫 Your Ticket Details")
                st.markdown(f"""
                <div class="result-card">
                    <h4>Class: {class_result['class_name']}</h4>
                    <p><strong>Description:</strong> {class_result['description']}</p>
                    <p><strong>Typical Passengers:</strong> {class_result['typical_passengers']}</p>
                    <p><strong>Fare (1912):</strong> £{fare_1912_gbp:.2f}</p>
                    <p><strong>Modern Equivalent:</strong> {symbol}{modern_fare['modern_value_original_currency']:,.2f} {currency}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("### 💰 What That Buys Today")
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                for comparison in modern_fare['comparisons']:
                    st.markdown(f"• {comparison}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_right:
                st.markdown("### 👤 Your Historical Match")
                st.markdown(f"""
                <div class="result-card">
                    <h4>{nearest['Name']}</h4>
                    <p><strong>Age:</strong> {nearest['Age']} | <strong>Sex:</strong> {nearest['Sex'].capitalize()}</p>
                    <p><strong>Class:</strong> {nearest['Pclass']} | <strong>Fare:</strong> £{nearest['Fare']:.2f}</p>
                    <p><strong>Embarked:</strong> {nearest['Embarked']}</p>
                    
                    <p style="margin-top: 1rem; padding: 0.5rem; background: #f0f0f0; border-radius: 5px;">
                        <strong>🎯 Your Prediction:</strong><br>
                        We used <strong>{nearest['Name']}'s</strong> travel details but replaced age, sex, and fare with yours!
                    </p>
                    
                    <p><strong>Historical Outcome:</strong> {nearest['Survived_text']}</p>
                    <p><strong>Match Confidence:</strong> {passenger_match['confidence']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("### 📈 Survival Analysis")
                st.markdown(f"""
                <div class="result-card">
                    <p><strong>Model Prediction:</strong> {survival_model_prob}% chance of survival</p>
                    <p><strong>Similar Passengers:</strong> {survival_prob}% survival rate</p>
                    <p><strong>Overall Rate:</strong> {(train_df['Survived'].mean() * 100):.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.exception(e)

else:
    st.markdown("""
    <div class="result-card">
        <h2 style="text-align: center;">⚓ Welcome Aboard! ⚓</h2>
        <p style="text-align: center; font-size: 1.2rem;">
            Enter your information in the sidebar to discover if you would have survived!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🌊 Titanic Facts")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**2,224** passengers aboard")
    with col2:
        st.info("**1,517** lives lost")
    with col3:
        st.info("**68%** mortality rate")