from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('saved_model.pkl')

# Mapping of numerical class to label
class_mapping = {
    0: 'Setosa',
    1: 'Versicolor',
    2: 'Virginica'
}

@app.route('/')
def home():
    return render_template('iris.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    input_features = [np.array(features)]
    prediction = model.predict(input_features)
    predicted_class = prediction[0]
    predicted_label = class_mapping.get(predicted_class, 'Unknown')
    return render_template('iris.html', prediction_text='{}'.format(predicted_label))

if __name__ == '__main__':
    app.run(debug=True)
