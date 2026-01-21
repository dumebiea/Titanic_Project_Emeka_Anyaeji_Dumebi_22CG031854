import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib
import os

def train_model():
    # Load dataset
    # url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    csv_path = "titanic.csv"
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Rename columns to standard names just in case
    df.rename(columns={
        'Siblings/Spouses Aboard': 'SibSp',
        'Parents/Children Aboard': 'Parch'
    }, inplace=True)

    # Feature Selection: Swapping Embarked for SibSp since Stanford dataset lacks Embarked
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'SibSp']
    target = 'Survived'
    
    # Verify columns exist
    missing_cols = [col for col in features if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns {missing_cols} in dataset. Available columns: {df.columns.tolist()}")
        # Fallback to dummy data creation or error
        return

    print(f"Selected features: {features}")
    
    X = df[features].copy()
    y = df[target]

    # Preprocessing
    # Handle missing values
    X['Age'] = X['Age'].fillna(X['Age'].median())
    X['Fare'] = X['Fare'].fillna(X['Fare'].median())
    
    # Encode categorical variables
    le_sex = LabelEncoder()
    X['Sex'] = le_sex.fit_transform(X['Sex'])
    
    # le_embarked removed as we are not using Embarked anymore

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Training (Random Forest)
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluation
    predictions = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    # Save Model
    model_path = os.path.join("model", "titanic_survival_model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Save encoders
    joblib.dump(le_sex, os.path.join("model", "le_sex.pkl"))
    # joblib.dump(le_embarked, os.path.join("model", "le_embarked.pkl")) 
    if os.path.exists(os.path.join("model", "le_embarked.pkl")):
        os.remove(os.path.join("model", "le_embarked.pkl"))
    print("Encoders saved.")

if __name__ == "__main__":
    # Ensure model directory exists
    if not os.path.exists("model"):
        os.makedirs("model")
    train_model()
