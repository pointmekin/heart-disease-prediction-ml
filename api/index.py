import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)
model = joblib.load('./models/model.joblib')

# set cors to allow all origins
from flask_cors import CORS
CORS(app)

# Define a simple route
@app.route('/')
def hello():
    return 'Heart disease prediction API using decision tree model'

# Define a route that accepts POST requests and returns JSON data
@app.route('/api/prediction', methods=['POST'])
def submit_data():
    # Assuming the request body is in JSON format
    data = request.get_json()
    # Process the data...

    # convert to dataframe
    df = pd.DataFrame(data, index=[0])

    print(type(df["general_health"][0]))

    # Return a response
    prediction = {
        'prediction': int(model.predict(df.iloc[0:1])[0]),
        'confidence': model.predict_proba(df.iloc[0:1])[0].tolist()  # Convert ndarray to list
    }
    print (prediction)
    return jsonify(prediction)
