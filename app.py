from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
from diagnostics import model_predictions, dataframe_summary, execution_time, missing_data_calculation, outdated_packages_list
from scoring import score_model
import json
import os



######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

prediction_model = None


#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    # Prepare test data
    inference_df = pd.read_csv(os.getcwd() + '/' + request.args.get('data_loc'))
    #call the prediction function
    preds = model_predictions(os.path.join(config['output_model_path']), inference_df)
    return preds

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def score():        
    #check the score of the deployed model
    f1_score = score_model(os.path.join(config['output_model_path']),
                           os.path.join(config['test_data_path']))
    return str(f1_score)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():        
    #check means, medians, and modes for each column
    stats = dataframe_summary(os.path.join(config['output_folder_path']) )
    return stats

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnose():        
    #check timing and percent NA values
    timings = execution_time()
    missing_data = missing_data_calculation(os.path.join(config['output_folder_path']))
    outdated = outdated_packages_list()

    ret = {
        'execution_time': timings,
        'missing_percentage': missing_data,
        'outdated_packages': outdated
    }

    return ret

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
