import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from train_model import train_and_evaluate_models, evaluate_feature_importance

# Streamlit App Configuration
st.set_page_config(page_title="House Price Prediction Application", page_icon="🏠", layout="wide")

st.title("🏠 House Price Prediction Project")
st.markdown("Upload a dataset to train machine learning models and predict house prices interactively.")

# 1. File Upload
uploaded_file = st.sidebar.file_uploader("Upload your Dataset (CSV format)", type=['csv'])

if uploaded_file is not None:
    # Read Dataset
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(df.head())
    
    # Select Target Variable
    target_col = st.sidebar.selectbox("Select Target Variable (e.g., Price)", df.columns)
    
    st.sidebar.markdown("---")
    
    if st.sidebar.button("Run ML Pipeline"):
        with st.spinner("Processing data, running Exploratory Data Analysis, and Training Models..."):
            
            # --- 3. Exploratory Data Analysis (EDA) ---
            st.header("Exploratory Data Analysis (EDA)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Target Variable Distribution")
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.histplot(df[target_col], kde=True, ax=ax, color='blue')
                ax.set_title(f"Distribution of {target_col}")
                st.pyplot(fig)
                
            with col2:
                st.subheader("Correlation Heatmap")
                numeric_df = df.select_dtypes(include=['int64', 'float64'])
                if not numeric_df.empty and len(numeric_df.columns) > 1:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
                    st.pyplot(fig)
                else:
                    st.info("Not enough numeric columns for a correlation heatmap.")
            
            # --- Data Preprocessing & Model Training (Steps 2, 4, 5, 6, 8) ---
            st.header("Model Training & Evaluation")
            results, best_name, best_model, y_test, y_pred, X_train = train_and_evaluate_models(df, target_col)
            
            # Display Evaluation Results
            results_df = pd.DataFrame(results).T
            st.dataframe(results_df.style.highlight_max(subset=['R2 Score'], color='lightgreen').highlight_min(subset=['RMSE', 'MAE', 'MSE'], color='lightgreen'))
            
            st.success(f"**Best Model Automatically Selected:** {best_name}")
            
            # --- 7. Visualizations (Actual vs Predicted & Feature Importance) ---
            st.header("Model Performance & Insights")
            
            col3, col4 = st.columns(2)
            
            with col3:
                st.subheader("Actual vs Predicted Values")
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, color='purple', ax=ax)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                ax.set_title(f"Actual vs Predicted ({best_name})")
                st.pyplot(fig)
                
            with col4:
                st.subheader("Feature Importance")
                importance_df = evaluate_feature_importance(best_model, X_train)
                if importance_df is not None:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.barplot(data=importance_df.head(10), x='Importance', y='Feature', ax=ax, palette='viridis')
                    ax.set_title("Top 10 Important Features")
                    st.pyplot(fig)
                else:
                    st.info(f"{best_name} does not support feature importance out-of-the-box (e.g., Linear Regression).")
            
            # Save the training columns in session state to use in prediction
            st.session_state['trained'] = True
            st.session_state['features'] = X_train.columns.tolist()
            st.session_state['dtypes'] = X_train.dtypes.to_dict()
            st.session_state['target_col'] = target_col

st.markdown("---")

# --- 9. Simple Prediction Interface ---
if st.session_state.get('trained', False):
    st.header("Interactive Prediction Interface")
    st.markdown(f"Enter house features to predict the **{st.session_state['target_col']}** using the trained best model.")
    
    # Dynamically create input fields based on features
    input_data = {}
    features = st.session_state['features']
    dtypes = st.session_state['dtypes']
    
    col1, col2, col3 = st.columns(3)
    
    for idx, feature in enumerate(features):
        with [col1, col2, col3][idx % 3]:
            if pd.api.types.is_numeric_dtype(dtypes[feature]):
                # Create number input for numeric features
                input_data[feature] = st.number_input(f"{feature}", value=0.0)
            else:
                # Create text input for categorical features
                input_data[feature] = st.text_input(f"{feature}", value="Unknown")
                
    if st.button("Predict Price"):
        # Load the best model
        model_path = os.path.join('models', 'best_model.pkl')
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                loaded_model = pickle.load(f)
            
            # Predict
            input_df = pd.DataFrame([input_data])
            try:
                prediction = loaded_model.predict(input_df)[0]
                st.success(f"### Predicted {st.session_state['target_col']}: {prediction:,.2f}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
        else:
            st.error("Model file not found. Please train the models first.")
else:
    st.info("Upload a dataset and run the ML pipeline to unlock the interactive prediction interface.")
