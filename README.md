# Disease Prediction and Doctor Recommendation System

-> Overview
This project is a Flask-based application that predicts diseases based on user-provided symptoms. Once a disease is predicted, 
the system suggests a specialist doctor and provides a brief description of the disease. 
It integrates multiple machine learning (ML) models to improve prediction accuracy and robustness.

**Project Structure**

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

**Key Features**
Symptom-Based Prediction: Predicts diseases based on symptoms using multiple ML models.
Doctor Recommendation: Recommends a specialist based on predicted disease.
Disease Description: Provides a brief description for each predicted disease.

**Machine Learning Models**
Uses an ensemble approach with multiple models to predict diseases:

Logistic Regression
Decision Tree
Random Forest
Support Vector Machine (SVM)
Naive Bayes
K-Nearest Neighbors (KNN)
Each model provides a prediction, and results are aggregated to determine the most likely disease.

**API Design**

Endpoint: POST /predict
Description: Receives symptoms and returns predicted diseases, confidence scores, recommended doctors, and disease descriptions.

->Input Format:
{
  "symptoms": ["chills", "knee_pain", "acidity"]
}

->Response Format:
  {
    "Disease": "Influenza",
    "Chances": 85.0,
    "Specialist": "General Physician",
    "Description": "A viral infection affecting the respiratory system."
  }

**Setup and Installation**

1. Clone the Repository
{
git clone https://github.com/yourusername/yourprojectname.git
cd yourprojectname
}

2.Create and Activate a Virtual Environment
{
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
}

3.Install Dependencies
{
pip install -r requirements.txt
}

4.Run the Application
{
python application.py
}

5. Access the Application: Open http://127.0.0.1:5000 in your browser.

**Future Enhancements**

Additional Symptoms and Diseases: Expanding the dataset for broader predictions.
Improved UI: A more interactive and user-friendly frontend.
Database Integration: Logging and analysis of user inputs and predictions.










