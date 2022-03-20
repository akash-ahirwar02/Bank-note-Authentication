# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib


app = Flask(__name__)

model = joblib.load("Bank_note_model.pkl")


@app.route('/')
def home():
    return render_template("index.html")



@app.route('/predict', methods =["POST"])
def predict():
    
        variance=request.form.get('variance')
        skewness=request.form.get('skewness')
        curtosis =request.form.get('curtosis')
        entropy=request.form.get('entropy')
        
        output = model.predict(pd.DataFrame(columns=['variance', 'skewness', 'curtosis', 'entropy'],
                                              data=np.array([variance,skewness,curtosis,entropy]).reshape(1,4)))[0]
        
        
        if(output==1):
            return render_template('index.html', prediction=" Authentic Note")
        
        else:
            return render_template('index.html', prediction="Fake Note")
           
    
    
    
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
