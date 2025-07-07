# Importing essential libraries and modules

from flask import Flask, render_template, request, Markup
import numpy as np
import pandas as pd

# import requests
import config
import pickle
import io
from PIL import Image

from testing_code import answer_checker

# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------

app = Flask(__name__)

# home page

@ app.route('/')
def home():
    title = 'Subjective Answers Evaluation Using Machine learning'
    return render_template('rust.html', title=title)

@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Subjective Answers Evaluation Using Deeplearning'
    
    if request.method == 'POST':
            input_text1 = request.form.get('inputText1')
            input_text2 = request.form.get('inputText2')
    

            prediction,per_cent = answer_checker(str(input_text1),str(input_text2))



            return render_template('rust-result.html', prediction=prediction,precaution="Similarity Score--",per_cent=str(per_cent)+str("%"),title=title)

# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=True)
