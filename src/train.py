import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from metadata import MODEL_PARAMS

def train_model(df: pd.DataFrame, target_column: str = "Exited") -> DecisionTreeClassifier:
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the model hyperparameters
    params = {"random_state": 8888,
          "min_samples_split": 3,
          "min_samples_leaf": 1}

    # Train the model
    model = DecisionTreeClassifier(**MODEL_PARAMS)
    model.fit(X_train, y_train)

    return model