import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Set path to current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'Credit_Card_Default.csv')

def train_and_save_model():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    # Preprocessing
    print("Preprocessing data...")
    # Mapping unknown categories (as done in the notebook)
    df['EDUCATION'] = df['EDUCATION'].replace([0, 5, 6], 4)
    df['MARRIAGE'] = df['MARRIAGE'].replace(0, 3)

    # Splitting Features and Target
    X = df.drop(['ID', 'default.payment.next.month'], axis=1)
    y = df['default.payment.next.month']

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Feature Scaling
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Model Building (Random Forest was the best in the notebook)
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Saving Model and Scaler
    print("Saving model and scaler...")
    joblib.dump(model, os.path.join(BASE_DIR, 'model.pkl'))
    joblib.dump(scaler, os.path.join(BASE_DIR, 'scaler.pkl'))
    print("Done! model.pkl and scaler.pkl created.")

if __name__ == "__main__":
    train_and_save_model()
