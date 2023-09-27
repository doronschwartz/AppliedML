import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Clean from Nan, form notebook
def cleaning(data):
    data = data.drop(columns=['gender'])
    data['experience'].fillna(0, inplace=True)
    data.loc[(data['last_new_job'].isnull()) & (data['company_type'].isnull()), 'last_new_job'] = 'never'
    data['enrolled_university'].fillna('no_enrollment', inplace=True)
    data.loc[(data['company_type'].isnull()) & (data['company_size'].isnull()), 'company_size'] = '0'
    data.dropna(subset=['last_new_job'], inplace=True)
    company_size_avg = data.groupby('company_type')['company_size'].transform(lambda x: x.mode().iloc[0] if not x.mode().empty else '0')
    data['company_size'].fillna(company_size_avg, inplace=True)
    data.dropna(subset=['last_new_job'], inplace=True)
    data['company_type'].fillna('None', inplace=True)
    return data

# Prepare the data to be all numeric
def preparation(data):
    data['company_size'] = data['company_size'].replace({
        '<10': 5,
        '10/49': 30,
        '50-99': 75,
        '100-500': 300,
        '500-999': 750,
        '1000-4999': 2500,
        '5000-9999': 7500,
        '10000+': 10000
    }).astype(int)
    data['city'] = data['city'].str.replace(r'^city_', '', regex=True)
    data['relevent_experience'] = data['relevent_experience'].map({'No relevent experience': 0, 'Has relevent experience': 1})
    data['last_new_job'] = data['last_new_job'].replace({'never': 0, '>4': 5}).astype(int)
    data['experience'] = data['experience'].replace({'<1': 0, '>20': 21}).astype(int)
    columns_to_encode = ['enrolled_university', 'major_discipline', 'company_type','education_level']
    data = pd.get_dummies(data, columns=columns_to_encode)
    return data


# Define the train_model function
def train_model(data):
    data = cleaning(data)
    data = preparation(data)

    # Split the data into features (X) and target (y)
    X = data.drop('target', axis=1)  # Features
    y = data['target']  # Target

    # Split the data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Standardize the feature data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    print(X_train.shape)
    # Have three layers, last one to get is the numeric prediction between one and zero
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_val, y_val), verbose=0)

    # Return it
    return model

# Define the inference function
def inference(trained_model, inference_data):
    # Clean and prepare the inference data
    inference_data = cleaning(inference_data)
    inference_data = preparation(inference_data)

    # Standardize the feature data
    scaler = StandardScaler()
    inference_data_scaled = scaler.fit_transform(inference_data)
    print("inference")
    print(inference_data_scaled.shape)
    # Make predictions using the trained model
    predictions_probabilities = trained_model.predict(inference_data_scaled)

    # Apply the threshold to get binary predictions (0 or 1)
    threshold = 0.5
    predictions = (predictions_probabilities > threshold).astype(int)

    #Get the employee IDs from the original inference data
    employee_ids = inference_data['enrollee_id']

    # Create a DataFrame to store the results
    results_df = pd.DataFrame({'enrollee_id': employee_ids, 'original_score': predictions_probabilities.flatten()})

    # Add "Yes" and "No" labels based on the .5 threshold
    results_df['prediction'] = predictions
    results_df['label'] = results_df['prediction'].apply(lambda x: 'Yes' if x else 'No')


    # Save the results to an output.csv file
    results_df.to_csv('output.csv', index=False)
    # Return the predictions
    return predictions

if __name__ == "__main__":
    # Training data
    data = pd.read_csv('csv/aug_train.csv')  # Replace with your data file path

    # Train the model
    trained_model = train_model(data)

    # Load the Test data for inference
    inference_data = pd.read_csv('csv/aug_test.csv')

    # Predicted CSV
    inference_predictions = inference(trained_model, inference_data)

    