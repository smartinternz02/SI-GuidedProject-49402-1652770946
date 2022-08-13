# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 22:27:42 2021

@author: Pujithas
"""
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
global graph
graph = tf.compat.v1.get_default_graph()



import pickle

with open(r'cv.pkl','rb') as file:
    cv=pickle.load(file)

import re 
import nltk
nltk.download("stopwords") 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

 
model = load_model("zomato_project.h5", compile = False)
app=Flask(__name__)
@app.route('/' )
def welcome():
    return render_template('home.html')
@app.route('/prediction', methods = ['GET','POST'])
def pred():
    if request.method == 'POST':
        review = request.form['message']
        review = re.sub('[^a-zA-Z]', ' ',review)
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word 
                  in set(stopwords.words('english'))]
        review = ' '.join(review)
        review = cv.transform([review]).toarray()
        
        y_p = model.predict(review)
        if y_p.argmax() == 0: 
            output = "Average"
        elif y_p.argmax() == 1:
            output = "Good"
        else:
            output = "Poor"
        return render_template('prediction.html',prediction = 
                               ("The Customer review is " + output)) 
    else:
        return render_template('prediction.html')
    
@app.route('/project')
def project():
    return render_template("project data.html")


if __name__ == "__main__":
     app.run(debug=True)