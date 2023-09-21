from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json


#################Function for model scoring
def score_model(model_folder, test_data_folder):
    #this function takes a trained model, load test data, and calculate an F1 score for the model relative to the test data
    with open(os.getcwd() + '/' + model_folder + '/trainedmodel.pkl', 'rb') as file:
        model = pickle.load(file)
    
    testdata = pd.read_csv(os.getcwd() + '/' + test_data_folder + '/testdata.csv')
    X = testdata.loc[:,['lastmonth_activity','lastyear_activity', 'number_of_employees']].values.reshape(-1, 3)
    y = testdata['exited'].values.reshape(-1,1)
    
    predicted = model.predict(X)
    f1_score = metrics.f1_score(predicted, y)
    
    #write the result to the latestscore.txt file
    metrics_file = open('latestscore.txt','w')
    metrics_file.write(str(f1_score))
    metrics_file.close()
    
    return f1_score

if __name__ == '__main__':
    ###################Load config.json and get path variables
    with open('config.json','r') as f:
        config = json.load(f) 
        
    model_path = os.path.join(config['output_model_path']) 
    test_data_path = os.path.join(config['test_data_path'])   
    
    f1_score = score_model(model_path, test_data_path)