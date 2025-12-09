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
    initial_sidebar_state="collapsed"
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
    /* MODIFIED: Changed background to dark blue-gray and set text to white */
    .result-card {
        background: #2a3a50; 
        color: white; /* Ensure text is readable */
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.4);
        margin: 1rem 0;
    }
    /* NEW: Vintage ticket styling (Request 4) */
    .vintage-ticket {
        background: #cba07b; /* Requested color */
        border: 5px solid #663300; /* Dark brown border */
        border-radius: 0; /* Sharp corners */
        padding: 2.5rem;
        box-shadow: 10px 10px 0px rgba(0,0,0,0.5); /* Drop shadow for depth */
        font-family: 'Times New Roman', serif; /* Classic font */
        color: #333; /* Dark text for contrast */
        position: relative;
        filter: url("#noise"); /* Add a subtle grainy effect */
    }
    .vintage-ticket::before {
        content: "RMS TITANIC";
        position: absolute;
        top: 5px;
        left: 50%;
        transform: translateX(-50%);
        font-size: 0.8rem;
        color: #663300;
        font-weight: bold;
        letter-spacing: 2px;
    }
    .vintage-ticket h4 {
        border-bottom: 2px dashed #996633;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
        text-align: center;
        color: #333;
    }
    .vintage-ticket strong {
        color: #663300;
    }
    /* MODIFIED: Removed character specific animations */
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
        /* Removed sink animation to keep it simpler with image */
    }
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-20px); }
    }
    /* Removed sink and character/wave/bubble animations */
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
    
    /* NEW: Underwater Spinner Styling (Request 1) */
    /* This targets the Streamlit spinner's container */
    .stSpinner {
        background: rgba(0, 70, 100, 0.95); /* Deep blue with transparency */
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        z-index: 9999;
        display: flex;
        align-items: center;
        justify-content: center;
        animation: water-pulse 5s infinite;
    }
    @keyframes water-pulse {
        0% { opacity: 0.9; }
        50% { opacity: 0.8; }
        100% { opacity: 0.9; }
    }
    .stSpinner > div > div { /* Targeting the inner text/icon */
        color: white !important;
        font-size: 2rem;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.5);
    }
</style>
<svg style="display: none;">
  <filter id="noise">
    <feTurbulence baseFrequency="0.6" type="fractalNoise" result="fractalNoise" />
    <feComponentTransfer>
      <feFuncA type="linear" slope="0.5" />
    </feComponentTransfer>
    <feComposite operator="in" in2="SourceGraphic" result="grain" />
  </filter>
</svg>
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

# Main content
# Define the predict_button variable first
predict_button = False 

# --- WELCOME BOX AND FACTS SHOWN BEFORE INPUTS ---
# Only show the welcome message and facts if the prediction hasn't been run yet (i.e., predict_button is False initially)
if not predict_button:
    st.markdown("""
    <div class="result-card">
        <h2 style="text-align: center; color: white;">⚓ Welcome Aboard! ⚓</h2>
        <p style="text-align: center; font-size: 1.2rem; color: white;">
            Fill out the form below and hit 'Predict My Fate!' to discover if you would have survived!
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

    st.markdown("---")


# --- INPUT FORM ---
st.markdown("### 📝 Enter Your Information")

col_age_sex, col_currency_earn = st.columns(2)

with col_age_sex:
    age = st.slider("Your Age", 1, 100, 25)
    sex = st.selectbox("Your Sex", ["Male", "Female"])

with col_currency_earn:
    currency = st.selectbox("Your Currency", ["USD", "GBP", "EUR", "INR"])
    annual_earnings = st.number_input(
        f"Annual Earnings ({currency})",
        min_value=1000,
        max_value=999999999999,
        value=50000,
        step=1000
    )

st.markdown("---")
# The button setting 'predict_button' must be here to trigger the logic below
predict_button = st.button("🎯 Predict My Fate!", use_container_width=True)
# --- END INPUT FORM ---


if predict_button:
    # MODIFIED: Custom spinner text/emoji for underwater feel (Request 1)
        
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
            
            
            # Display results
            st.markdown("---")
            
            # MODIFIED: Animation with EMOJIS (to replace text placeholder)
            if prediction == 1:
                st.markdown(f"""
                <div class="survived-animation">
                    <div style="font-size: 5rem; margin-bottom: 1rem;">🚢✅</div>
                    <h1 style="color: white;">🎉 YOU SURVIVED! 🎉</h1>
                    <h3 style="color: white;">The iceberg was successfully avoided!</h3>
                    <p style="color: white; font-size: 1.2rem;">
                        Your journey sailed safely across the Atlantic
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
                    <div style="font-size: 5rem; margin-bottom: 1rem;">🚢💥🧊</div>
                    <h1 style="color: #ff6b6b;">💔 YOU DIDN'T SURVIVE 💔</h1>
                    <h3 style="color: #ffaaaa;">Lost in the Atlantic...</h3>
                    <p style="color: #cccccc; font-size: 1.2rem;">
                        You went down with the ship after impact
                    </p>
                    <div style="margin-top: 2rem;">
                        <span style="font-size: 3rem;">🧊</span>
                        <span style="font-size: 2rem;">💥</span>
                        <span style="font-size: 3rem;">🌊</span>
                        <span style="font-size: 2rem;">🚢</span>
                        <span style="font-size: 3rem;">🌊</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Stats
            st.markdown("### 📊 Your Titanic Profile")
            col1, col2, col3, col4 = st.columns(4)
            
            symbols = {'USD': '$', 'GBP': '£', 'EUR': '€', 'INR': '₹'}
            symbol = symbols.get(currency, '')
            
            # MODIFIED: Swapped stat-label and stat-number
            with col1:
                st.markdown(f'<div class="stat-box"><div class="stat-label">Passenger Class</div><div class="stat-number">{pclass}</div></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="stat-box"><div class="stat-label">Survival Probability</div><div class="stat-number">{survival_model_prob}%</div></div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="stat-box"><div class="stat-label">Modern Ticket Price</div><div class="stat-number">{symbol}{modern_fare["modern_value_original_currency"]:,.0f}</div></div>', unsafe_allow_html=True)
            with col4:
                st.markdown(f'<div class="stat-box"><div class="stat-label">1912 Fare</div><div class="stat-number">£{fare_1912_gbp:.0f}</div></div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Details
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.markdown("### 🎫 Your Ticket Details")
                # MODIFIED: Applied vintage-ticket class
                st.markdown(f"""
                <div class="vintage-ticket">
                    <h4>PASSAGE TICKET - {class_result['class_name']}</h4>
                    <p><strong>Description:</strong> {class_result['description']}</p>
                    <p><strong>Typical Passengers:</strong> {class_result['typical_passengers']}</p>
                    <p><strong>Fare (1912):</strong> £{fare_1912_gbp:.2f}</p>
                    <p><strong>Modern Equivalent:</strong> {symbol}{modern_fare['modern_value_original_currency']:,.2f} {currency}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("### 💰 What That Buys Today")
                
                # FIX: Bullet points embedded inside the result-card markdown block (Request 2)
                comparison_list_html = "".join([f"<li>{c}</li>" for c in modern_fare['comparisons']])
                st.markdown(f"""
                <div class="result-card">
                    <ul style="padding-left: 1.5rem;">
                        {comparison_list_html}
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with col_right:
                st.markdown("### 👤 Your Historical Match")
                st.markdown(f"""
                <div class="result-card">
                    <h4>{nearest['Name']}</h4>
                    <p><strong>Age:</strong> {nearest['Age']} | <strong>Sex:</strong> {nearest['Sex'].capitalize()}</p>
                    <p><strong>Class:</strong> {nearest['Pclass']} | <strong>Fare:</strong> £{nearest['Fare']:.2f}</p>
                    <p><strong>Embarked:</strong> {nearest['Embarked']}</p>
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