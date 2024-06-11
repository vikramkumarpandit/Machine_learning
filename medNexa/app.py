from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter

app = Flask(__name__)

# Load symptom data
dis_sym_data = pd.read_csv("Original_Dataset.csv")
columns_to_check = [col for col in dis_sym_data.columns if col != 'Disease']

symptoms = dis_sym_data.iloc[:, 1:].values.flatten()
symptoms = list(set(symptoms))

# Create a list to store the data frames for each symptom
symptom_dfs = []

for symptom in symptoms:
    # Create a DataFrame with a single column for the current symptom
    symptom_df = pd.DataFrame({symptom: dis_sym_data.iloc[:, 1:].apply(lambda row: int(symptom in row.values), axis=1)})
    symptom_dfs.append(symptom_df)

# Concatenate all symptom DataFrames along the columns axis
dis_sym_data_v1 = pd.concat(symptom_dfs, axis=1)

# Convert column names to strings
dis_sym_data_v1.columns = dis_sym_data_v1.columns.astype(str)

# Label encoding for 'Disease' column
le = LabelEncoder()
dis_sym_data_v1['Disease'] = le.fit_transform(dis_sym_data['Disease'])

X = dis_sym_data_v1.drop(columns="Disease")
y = dis_sym_data_v1['Disease']

# Define ML algorithms
algorithms = {
    'Logistic Regression': {"model": LogisticRegression()},
    'Decision Tree': {"model": tree.DecisionTreeClassifier()},
    'Random Forest': {"model": RandomForestClassifier()},
    'SVM': {"model": svm.SVC(probability=True)},
    'NaiveBayes': {"model": GaussianNB()},
    'K-Nearest Neighbors': {"model": KNeighborsClassifier()},
}

# Fit the models using the training data
for model_name, values in algorithms.items():
    model = values["model"]
    model.fit(X, y)

# Load doctor versus disease data
doc_data = pd.read_csv("Doctor_Versus_Disease.csv", encoding='latin1', names=['Disease', 'Specialist'])
doc_data['Specialist'] = np.where((doc_data['Disease'] == 'Tuberculosis'), 'Pulmonologist', doc_data['Specialist'])

# Load disease description data
des_data = pd.read_csv("Disease_Description.csv", encoding='latin1')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/enter_symptoms', methods=['POST'])
def enter_symptoms():
    if request.method == 'POST':
        num_symptoms = int(request.form['num_symptoms'])
        return render_template('enter_symptoms.html', num_symptoms=num_symptoms, all_symptoms=symptoms)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        num_symptoms = int(request.form['num_symptoms'])
        input_symptoms = [request.form[f'symptom{i+1}'] for i in range(num_symptoms)]
        result_df = test_input(input_symptoms, algorithms, le, doc_data, des_data)
        return jsonify(result_df.to_dict(orient='records'))

def test_input(symptoms, algorithms, le, doc_data, des_data):
    test_data = {symptom: 1 if symptom in symptoms else 0 for symptom in X.columns}
    test_df = pd.DataFrame([test_data])

    predicted = []
    for model_name, values in algorithms.items():
        model = values["model"]
        predict_disease = model.predict(test_df)
        predict_disease = le.inverse_transform(predict_disease)
        predicted.extend(predict_disease)

    disease_counts = Counter(predicted)
    num_algorithms = len(algorithms)
    percentage_per_disease = {disease: (count / num_algorithms) * 100 for disease, count in disease_counts.items()}

    result_df = pd.DataFrame({"Disease": list(percentage_per_disease.keys()),
                              "Chances": list(percentage_per_disease.values())})
    result_df = result_df.merge(doc_data, on='Disease', how='left')
    result_df = result_df.merge(des_data, on='Disease', how='left')
    return result_df

if __name__ == '__main__':
    app.run(debug=True)
