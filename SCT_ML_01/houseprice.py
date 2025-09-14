import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load datasets
train_data = pd.read_csv(r"C:\Users\gopik\MLProjects\house-prices\train.csv")
test_data = pd.read_csv(r"C:\Users\gopik\MLProjects\house-prices\test.csv")

# Select features
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']

X_train = train_data[features]
y_train = train_data['SalePrice']

X_test = test_data[features]

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
test_predictions = model.predict(X_test)

# Combine features with predictions for display
results = pd.DataFrame({
    "GrLivArea": X_test["GrLivArea"],
    "BedroomAbvGr": X_test["BedroomAbvGr"],
    "FullBath": X_test["FullBath"],
    "PredictedPrice": test_predictions
})

# Print few rows in terminal
print("📊 Predicted House Prices (showing first 10):")
print(results.head(10))

# Save predictions in Kaggle submission format
submission = pd.DataFrame({
    "Id": test_data["Id"],
    "SalePrice": test_predictions
})
submission.to_csv("submission.csv", index=False)

print("\n✅ Predictions saved to submission.csv")
