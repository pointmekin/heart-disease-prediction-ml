import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)
model = joblib.load('./model.joblib')

@app.route('/')
def home():
    return 'Hello, World!'

@app.route('/about')
def about():
    return 'About'