from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json


#################Function for training the model
def train_model(input_data_path, output_model_path):    
    #Logistic regression for training
    logit = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
    
    #fit the logistic regression to data
    trainingdata = pd.read_csv(os.getcwd() + '/' + input_data_path + '/finaldata.csv')
    # Reshape dataframe features as arrays
    X = trainingdata.loc[:,['lastmonth_activity','lastyear_activity', 'number_of_employees']].values.reshape(-1, 3)
    y = trainingdata['exited'].values.reshape(-1, 1).ravel()
    model = logit.fit(X, y)
    
    #write the trained model to workspace in a file called trainedmodel.pkl
    pickle.dump(model, open(os.getcwd() + '/' + output_model_path +  '/trainedmodel.pkl', 'wb'))
    
    
if __name__ == '__main__':
    ###################Load config.json and get path variables
    with open('config.json','r') as f:
        config = json.load(f) 

    dataset_csv_path = os.path.join(config['output_folder_path']) 
    model_path = os.path.join(config['output_model_path']) 
    
    train_model(dataset_csv_path, model_path)