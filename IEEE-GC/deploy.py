from google.cloud import aiplatform
import json
import pandas as pd


project_id = "big-celerity-404704"
location = "us-east4"
endpoint_id = "6739469716592525312"

#Local Machine Path for Transaction Data
original_row = pd.read_csv('/Users/doronschwartz/YCCS/AppliedML/IEEE-GC/data/train_transaction.csv', nrows=1)

# Convert row to dictionary for JSON purposes
original_data = original_row.iloc[0].astype(str).to_dict()

# Initialize AI Platform
aiplatform.init(project=project_id, location=location)

# Get the endpoint
endpoint = aiplatform.Endpoint(endpoint_name=f"projects/{project_id}/locations/{location}/endpoints/{endpoint_id}")

# Prompt A user for input for the below fields
transaction_id = input("Enter TransactionId: ")
product_cd = input("Enter ProductCD: ")
transaction_amt = input("Enter TransactionAmt: ")
card4 = input("Enter Card Type (card4): ")
card6 = input("Enter Card Category (card6): ")
p_emaildomain = input("Enter P_emaildomain: ")

# Update original_data with user input
original_data['TransactionId'] = transaction_id
original_data['ProductCD'] = product_cd
original_data['TransactionAmt'] = transaction_amt
original_data['card4'] = card4
original_data['card6'] = card6
original_data['P_emaildomain'] = p_emaildomain

# Convert input_data to JSON format
json_data = json.dumps({'instances': [original_data]})

# Make a prediction
response = endpoint.predict(instances=[original_data])

print("Prediction Response:")
# Extract prediction scores
prediction_scores = response.predictions[0]['scores']
class_labels = response.predictions[0]['classes']

# Print prediction scores
print("Prediction Scores:")
for label, score in zip(class_labels, prediction_scores):
    print(f"{label}: {score}")

# Interpret prediction
predicted_class = class_labels[prediction_scores.index(max(prediction_scores))]


# Print interpretation
if predicted_class == '1':
    print(" Likely Fraudulent Transaction")
else:
    print("Likely Non-Fraudulent Transaction")