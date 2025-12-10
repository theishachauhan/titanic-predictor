# 🚢 Titanic Survival Predictor - Would YOU Survive?

An interactive web application that predicts whether you would have survived the Titanic disaster based on your age, sex, and annual earnings. Travel back to 1912 and discover your fate!

![Titanic Banner](https://a-z-animals.com/media/2022/12/shutterstock_2143164935.jpg)

## 🌟 Features

- **💰 Currency Conversion**: Converts your modern earnings to 1912 British Pounds using live exchange rates
- **🎫 Class Assignment**: Automatically determines which passenger class (1st, 2nd, or 3rd) you would travel in
- **👤 Historical Matching**: Finds the closest real Titanic passenger to you based on age, sex, and class
- **🤖 ML Prediction**: Uses a trained Decision Tree model to predict your survival probability
- **💵 Modern Fare Calculator**: Shows what your ticket would cost in today's money with real-world comparisons
- **🎨 Interactive Animations**: 
  - Floating to New York if you survive 🗽
  - Sinking to the ocean floor if you don't 💧
  - Character selection based on age/sex

**Try it now:** [Titanic Predictor Live App](https://your-app-url.streamlit.app)


## 🛠️ Technologies Used

- **Frontend**: Streamlit
- **Machine Learning**: scikit-learn (Decision Tree Classifier with Grid Search)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **APIs**: Frankfurter API (live currency exchange rates)
- **Deployment**: Streamlit Cloud & Docker

## Model Performance

- **Algorithm**: Decision Tree Classifier (optimized with GridSearchCV)
- **Training Accuracy**: ~82-85%
- **Test Accuracy**: ~80-82%
- **Features Used**: Age, Sex, Pclass, Fare, Family Size, Embarkation Port, and one-hot encoded Name/Cabin features

## Quick Start

### Prerequisites

- Python 3.10 or higher

### Local Installation

> **Clone the repository**
```bash
git clone https://github.com/theishachauhan/titanic-predictor.git
cd titanic-predictor
```
> **To Train the model** 
```bash
python models/titanic-survival.py
```
> **To Run the application**
```bash
streamlit run app.py
```
> **Open the browser**
```
http://localhost:8501
```

## 📁 Project Structure

```
titanic-predictor/
│
├── app.py                      # Main Streamlit application
├── titanic-survival.py         # Model training script
├── requirements.txt            # Python dependencies
│
├── data/
│   └── train.csv              # Titanic dataset from Kaggle
│
├── models/
│   └── titanic_model.pkl      # Trained ML model
│
├── utils/
│   ├── __init__.py
│   ├── currency_converter.py  # Modern to 1912 currency conversion
│   ├── class_classifier.py    # Passenger class determination
│   ├── passenger_finder.py    # Nearest passenger matching
│   └── fare_calculator.py     # 1912 to modern fare conversion
│
├── Dockerfile                  # Docker containerization
├── docker-compose.yml          # Docker Compose configuration
├── kubernetes/                 # Kubernetes deployment files
│   ├── deployment.yaml
│   ├── service.yaml
│   └── ingress.yaml
│
└── README.md                   # This file
```

## 🎮 How It Works

### 1. User Input
- Enter your age, sex, and annual earnings in your local currency

### 2. Currency Conversion
- Fetches live exchange rates from Frankfurter API
- Converts modern earnings to 1912 British Pounds
- Applies historical inflation adjustments (130x multiplier for GBP)
- Estimates Titanic fare as ~4% of annual income

### 3. Class Assignment
- **1st Class**: Fare ≥ £30 (luxury accommodations)
- **2nd Class**: Fare £13-30 (middle-class comfort)
- **3rd Class**: Fare < £13 (basic steerage)

### 4. Passenger Matching
- Uses Euclidean distance on normalized features (Age, Sex, Pclass)
- Finds the closest historical passenger from the dataset
- Inherits their travel details (family, cabin, embarkation port)

### 5. Prediction
- Creates a hybrid profile: user's age/sex/fare + matched passenger's context
- One-hot encodes all features to match training format
- Predicts survival using trained Decision Tree model
- Returns probability percentage

### 6. Modern Fare Calculation
- Converts 1912 fare back to modern currency
- Provides contextual comparisons (e.g., "cost of a MacBook Pro")

## 🐳 Docker Deployment

### Build and Run with Docker

```bash
# Build the image
docker build -t titanic-predictor .

# Run the container
docker run -p 8501:8501 titanic-predictor

# Open browser at http://localhost:8501
```

## 📈 Model Training Details

The model is trained using scikit-learn's `DecisionTreeClassifier` with hyperparameter tuning.

## 🌐 API Endpoints

The app uses the following external APIs:

- **Currency Exchange**: `https://api.frankfurter.app/latest`
  - Free, no API key required
  - Updates daily
  - Supports 30+ currencies but i have only used 4 in this project.

## 📊 Dataset Information

- **Source**: [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Kaggle** for the Titanic dataset
- **RMS Titanic Historical Society** for historical fare data
- **Frankfurter API** for free currency exchange rates
- **Streamlit** for the amazing web framework
- Inspired by the 1997 film *Titanic* directed by James Cameron

## 📧 Contact
cishasingh@gmail.com

**Project Link**: [https://github.com/theishachauhan/titanic-predictor](https://github.com/theishachauhan/titanic-predictor)

---

## 🎯 Future Enhancements

- [ ] Add support for family groups (traveling with family members)
- [ ] Include cabin location visualization
- [ ] Historical passenger stories and photos
- [ ] Multi-language support and better animations
- [ ] Save and share prediction results
- [ ] Add XGBoost models for comparison
- [ ] Add historical timeline of the Titanic journey


---

**⭐ If you found this project interesting, please give it a star!**
