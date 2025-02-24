# Customer Churn Prediction Using Machine Learning

## Overview
This project focuses on predicting **customer churn** using machine learning techniques. Churn prediction is crucial for businesses to retain customers by identifying those who are likely to leave.

## Features of the Project
- **Data Preprocessing**: Handling missing values, encoding categorical variables, and feature scaling.
- **Multiple Models**: Implemented **Random Forest, SVM, and XGBoost** for predictions.
- **Hyperparameter Tuning**: Used GridSearchCV to optimize the Random Forest model.
- **Model Evaluation**: Accuracy score, classification report, and confusion matrix.
- **Feature Importance Analysis**: Identified the most critical features affecting churn.

## Dataset
The dataset used is `customer_churn.csv`, which contains customer details and churn labels.

## Installation
Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

## Training the Models
Run the script to train the models and save them as `.pkl` files:
```bash
python train.py
```

## Making Predictions
You can load the saved models and use them for predictions:
```python
import joblib
import numpy as np

rf_model = joblib.load("rf_model.pkl")
input_data = np.array([feature1, feature2, feature3]).reshape(1, -1)
result = rf_model.predict(input_data)
print("Prediction:", "Churn" if result[0] == 1 else "Not Churn")
```

## Model Evaluation
Evaluate model performance using:
```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = rf_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
```

## Contribution
Feel free to contribute by improving model performance or adding new features.

## License
This project is open-source under the MIT License.

---
This README provides an easy-to-follow guide to understanding and using your project. Let me know if you need any modifications! 🚀

