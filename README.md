# Disease Prediction and Doctor Recommendation System

## Overview

This project is a Flask-based web application that predicts potential diseases based on symptoms provided by the user. After identifying the possible disease, the application recommends a specialist and provides a description of the condition. The system uses an ensemble of machine learning models for reliable predictions, drawing on data from an internal dataset.

---

## Key Features

- **Symptom-Based Disease Prediction**: Predicts diseases using symptoms input by the user through an ensemble of machine learning models.
- **Doctor Recommendation**: Provides a recommendation for a relevant specialist doctor based on the predicted disease.
- **Disease Description**: Includes brief descriptions of predicted diseases for user education and awareness.

---

## Machine Learning Model Design

The system employs multiple machine learning models to enhance accuracy. These models are trained on a dataset with symptom-disease mappings:

- **Logistic Regression**: Used for its interpretability and efficiency on smaller datasets.
- **Decision Tree**: Provides a simple decision-making framework, easy to visualize and understand.
- **Random Forest**: An ensemble of decision trees to improve accuracy and reduce overfitting.
- **Support Vector Machine (SVM)**: Effective for high-dimensional space classification.
- **Naive Bayes**: A probabilistic model suitable for categorical data.
- **K-Nearest Neighbors (KNN)**: A straightforward, non-parametric model for classification.

Each model contributes its prediction, and the final disease prediction is derived from a consensus of these models.

---

## API Design

### Endpoint: `POST /predict`

- **Description**: Receives symptoms in JSON format and returns potential diseases with prediction confidence, a recommended specialist, and a disease description.
- **Request Format**:
    ```json
    {
      "symptoms": ["chills", "knee_pain", "acidity"]
    }
    ```
- **Response Format**:
    ```json
    [
      {
        "Disease": "Influenza",
        "Chances": 85.0,
        "Specialist": "General Physician",
        "Description": "A viral infection affecting the respiratory system."
      },
      ...
    ]
    ```
- **Example**:
    - **Request**:
      ```json
      {
        "symptoms": ["fever", "headache", "nausea"]
      }
      ```
    - **Response**:
      ```json
      [
        {
          "Disease": "Malaria",
          "Chances": 78.5,
          "Specialist": "Infectious Disease Specialist",
          "Description": "A disease caused by a plasmodium parasite, transmitted by mosquito bites."
        }
      ]
      ```

---

## Project Structure

```plaintext
.
├── application.py            # Main Flask application
├── templates/
│   ├── index.html            # Home page template
│   ├── enter_symptoms.html   # Symptom input form template
│   └── result.html           # Prediction result display template
├── Original_Dataset.csv      # Symptom-disease dataset
├── Doctor_Versus_Disease.csv # Disease-specialist mapping
├── Disease_Description.csv   # Disease description data
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation















