# House Price Prediction

A complete Machine Learning project for predicting house prices.

## Features
- Upload a dataset (CSV file) via a simple Streamlit interface
- Explore complete Data Preprocessing (handles missing values, encodes categorical variables, normalizes numerical data)
- Perform Exploratory Data Analysis (Correlation Healthmap, Distribution Plots, Feature Importance)
- Train 4 powerful Machine Learning models at once:
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
  - Gradient Boosting Regressor
- Evaluate models using MAE, MSE, RMSE, R² Score
- Automatically select the best model
- Interactive predicting interface to get new price predictions seamlessly

## Directory Structure
```
house_price_prediction/
│
├── data/              # Stores the datasets
├── notebooks/         # For exploratory jupyter notebooks
├── models/            # Saved `.pkl` model files are kept here
├── app.py             # Streamlit application
├── train_model.py     # Main ML pipeline, training and evaluation logic
├── requirements.txt   # Dependencies
└── README.md          # Project instructions
```

## How to Run:
1. Ensure you have Python installed securely.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```
4. Access the web app in your browser, upload your dataset, and follow along!
