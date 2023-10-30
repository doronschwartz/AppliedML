import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

#Doron Schwartz, Week 7, 10/30/23
def load_data():
    data_folder = "/Users/doronschwartz/YCCS/Schwartz_Doron_800590794/AppliedML/IEE/data"
    train_transaction = pd.read_csv(data_folder + "/train_transaction.csv")
    train_identity = pd.read_csv(data_folder + "/train_transaction.csv")
    return train_transaction, train_identity

def train_and_evaluate_models(X_train, y_train, X_val, y_val, X_test, y_test):
    # Neural Network
    model_nn = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model_nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model_nn.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_val, y_val))
    
    # Make probability predictions on the test data
    y_pred_nn = model_nn.predict(X_test)
    roc_auc_nn = roc_auc_score(y_test, y_pred_nn)
    print("Neural Network ROC-AUC Score: {:.4f}".format(roc_auc_nn))
    
    # XGBoost
    model_xgb = xgb.XGBClassifier()
    model_xgb.fit(X_train, y_train)
    y_pred_xgb = model_xgb.predict_proba(X_test)[:, 1]
    roc_auc_xgb = roc_auc_score(y_test, y_pred_xgb)
    print("XGBoost ROC-AUC Score: {:.4f}".format(roc_auc_xgb))
    
    # CatBoost
    model_cat = CatBoostClassifier(iterations=1000, depth=6, learning_rate=0.1, loss_function='Logloss', verbose=100)
    model_cat.fit(X_train, y_train, eval_set=(X_val, y_val))
    y_pred_cat = model_cat.predict_proba(X_test)[:, 1]
    roc_auc_cat = roc_auc_score(y_test, y_pred_cat)
    print("CatBoost ROC-AUC Score: {:.4f}".format(roc_auc_cat))

    # Random Forest
    model_rf = RandomForestClassifier(random_state=42)
    model_rf.fit(X_train, y_train)
    y_pred_rf = model_rf.predict_proba(X_test)[:, 1]
    roc_auc_rf = roc_auc_score(y_test, y_pred_rf)
    print("Random Forest ROC-AUC Score: {:.4f}".format(roc_auc_rf))

    # Save the trained models
    joblib.dump(model_nn, 'model_nn.pkl')
    joblib.dump(model_xgb, 'model_xgb.pkl')
    joblib.dump(model_cat, 'model_cat.pkl')
    joblib.dump(model_rf, 'model_rf.pkl')

def main():
    train_transaction, train_identity = load_data()
    
    # Select columns for modeling
    selected_columns = ['ProductCD', 'TransactionAmt', 'card1', 'card2', 'card3', 'card4',
                        'card5', 'card6', 'P_emaildomain', 'isFraud', 'V257', 'V246', 'V244',
                        'V242', 'V201', 'V200', 'V189', 'V188', 'V258', 'V45', 'V158', 'V156',
                        'V149', 'V228', 'V44', 'V86', 'V87', 'V170', 'V147', 'V52']

    # Extract relevant columns
    data = train_transaction[selected_columns]

    # One-hot encode non-numeric columns
    non_numeric_cols = data.select_dtypes(exclude=['int64', 'float64']).columns
    data = pd.get_dummies(data, columns=non_numeric_cols)

    # Split the data into features (X) and target (y)
    X = data.drop(columns=['isFraud'])
    y = data['isFraud']

    imputer = SimpleImputer(strategy='mean')  # Use mean strategy for imputation
    X = imputer.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train and evaluate the models
    train_and_evaluate_models(X_train, y_train, X_val, y_val, X_test, y_test)

if __name__ == "__main__":
    main()
