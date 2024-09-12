import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.metrics import mean_squared_error


def random_date(start, end):
    return start + timedelta(seconds=random.randint(0, int((end - start).total_seconds())))

def create_dummy_dataset(num_rows, num_categories, date_start, date_end):
    date_start = datetime.strptime(date_start, '%Y-%m-%d')
    date_end = datetime.strptime(date_end, '%Y-%m-%d')
    
   
    dates = [random_date(date_start, date_end) for _ in range(num_rows)]
    

    categorical_data = {
        f'Category_{i+1}': [f'Value_{random.randint(1, num_categories)}' for _ in range(num_rows)]
        for i in range(10)  # 10 categorical columns
    }
    
    categorical_data['Date'] = dates
   
    categorical_data['Target'] = np.random.rand(num_rows) * 100  # Continuous values between 0 and 100
    
    df = pd.DataFrame(categorical_data)
    return df


def preprocess_data(df):
    
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Hour'] = df['Date'].dt.hour
    
  
    df = df.drop('Date', axis=1)
    
    label_encoders = {}
    for col in df.columns:
        if df[col].dtype == 'object':  # Categorical columns
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le  # Store the label encoder if needed later
    
    return df, label_encoders


def apply_lightgbm_regression(df):
  
    X = df.drop('Target', axis=1)
    y = df['Target']
    
   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
   
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test)
    
   
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9
    }
    
    
    lgb_model = lgb.train(params, train_data, valid_sets=[test_data], num_boost_round=100)
    
   
    y_pred = lgb_model.predict(X_test)
    
    
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse:.4f}')
    
    return lgb_model


df = create_dummy_dataset(num_rows=1000, num_categories=5, date_start='2023-01-01', date_end='2024-01-01')

df_processed, label_encoders = preprocess_data(df)

model = apply_lightgbm_regression(df_processed)
