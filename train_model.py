import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os

def create_preprocessor(X):
    """
    Creates a scikit-learn ColumnTransformer that preprocessing numerical
    and categorical features.
    Handles missing values, standardizes numerical data, and one-hot encodes categoricals.
    """
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def train_and_evaluate_models(df, target_col):
    """
    Splits data, preprocesses it, and trains multiple models.
    Returns the evaluation metrics, the best model pipeline, actual vs pred on test set,
    and the name of the best model.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in the dataset.")
        
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    preprocessor = create_preprocessor(X)
    
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    best_model = None
    best_score = -float('inf')
    best_name = ""
    best_predictions = None
    
    trained_pipelines = {}
    
    for name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('model', model)])
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Predict on test set
        y_pred = pipeline.predict(X_test)
        
        # Evaluate
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2 Score": r2
        }
        
        trained_pipelines[name] = pipeline
        
        # Automatic Best Model Selection based on R2 Score
        if r2 > best_score:
            best_score = r2
            best_model = pipeline
            best_name = name
            best_predictions = y_pred

    # Save the best model
    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', 'best_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)

    return results, best_name, best_model, y_test.values, best_predictions, X_train

def evaluate_feature_importance(best_model, X_train):
    """
    Extracts feature importances if the model supports it.
    Works for Tree-based models.
    """
    try:
        # Extract the named steps
        model_step = best_model.named_steps['model']
        preprocessor_step = best_model.named_steps['preprocessor']
        
        if hasattr(model_step, 'feature_importances_'):
            # Get feature names from preprocessor
            numeric_features = preprocessor_step.transformers_[0][2]
            categorical_features = preprocessor_step.transformers_[1][1].named_steps['onehot'].get_feature_names_out(preprocessor_step.transformers_[1][2])
            
            all_features = list(numeric_features) + list(categorical_features)
            importances = model_step.feature_importances_
            
            importance_df = pd.DataFrame({
                'Feature': all_features,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)
            
            return importance_df
        else:
            return None
    except Exception as e:
        print(f"Could not extract feature importances: {e}")
        return None
