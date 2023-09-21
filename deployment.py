from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
import shutil
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json


####################function for deployment
def store_model_into_pickle(model_folder, prod_folder):
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    #with open(os.getcwd() + '/' + model_folder + '/trainedmodel.pkl', 'rb')as file:
    #    model = pickle.load(file)
    #pickle.dump(model, open(os.getcwd() + '/' + prod_folder +  '/trainedmodel.pkl', 'wb'))
    shutil.copy2(os.getcwd() + '/' + model_folder + '/trainedmodel.pkl', os.getcwd() + '/' + prod_folder +  '/trainedmodel.pkl')
    shutil.copy2(os.getcwd() + '/latestscore.txt', os.getcwd() + '/' + prod_folder +  '/latestscore.txt')
    shutil.copy2(os.getcwd() + '/ingestedfiles.txt', os.getcwd() + '/' + prod_folder +  '/ingestedfiles.txt')
    
        
        
if __name__ == '__main__':
    ###################Load config.json and get path variables
    with open('config.json','r') as f:
        config = json.load(f) 

    model_path = os.path.join(config['output_model_path']) 
    prod_deployment_path = os.path.join(config['prod_deployment_path']) 
    
    store_model_into_pickle(model_path, prod_deployment_path)