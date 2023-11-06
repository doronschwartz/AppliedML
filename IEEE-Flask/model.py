import numpy as np
import pandas as pd
import xgboost as xgb
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

current_directory = "/Users/doronschwartz/YCCS/AppliedML/IEEE-Flask"

# Local function to load downloaded data
def load_data():
    data_folder = "/Users/doronschwartz/YCCS/AppliedML/IEEE-Flask/data"
    train_transaction = pd.read_csv(data_folder + "/train_transaction.csv")
    train_identity = pd.read_csv(data_folder + "/train_transaction.csv")
    return train_transaction, train_identity


def train_and_evaluate_model(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
     # Scale the features for better perfomance
    scaler = StandardScaler(with_mean=False)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Create and train the XGBoost model
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred)
    print("XGBoost ROC-AUC Score: {:.4f}".format(roc_auc))

    model_filename = os.path.join(current_directory, 'xgboost_model.pkl')
    # Save the trained XGBoost model
    joblib.dump(model, model_filename)

def main():
    train_transaction, train_identity = load_data()
    
    # Select columns for modeling
    selected_columns = ['ProductCD', 'TransactionAmt', 'card1', 'card2', 'card3', 'card4',
                        'card5', 'card6', 'P_emaildomain', 'isFraud', 'V257', 'V246', 'V244',
                        'V242', 'V201', 'V200', 'V189', 'V188', 'V258', 'V45', 'V158', 'V156',
                        'V149', 'V228', 'V44', 'V86', 'V87', 'V170', 'V147', 'V52']

    # Limit the dataframe to the selected columns
    data = train_transaction[selected_columns]

    X = data.drop(columns=['isFraud'])
    y = data['isFraud']
    
    
    non_numeric_cols = X.select_dtypes(exclude=['int64', 'float64']).columns


    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer,non_numeric_cols)
        ])
    
    X = preprocessor.fit_transform(X)

    imputer = SimpleImputer(strategy='mean')  # Use mean strategy for imputation
    X = imputer.fit_transform(X)
    
    preprocessor_filename = os.path.join(current_directory, 'preprocessor.pkl')
    # Save the preprocessor (transformer) using joblib
    joblib.dump(preprocessor, preprocessor_filename)

    # Train and evaluate the XGBoost model
    train_and_evaluate_model(X, y)

if __name__ == "__main__":
    main()
