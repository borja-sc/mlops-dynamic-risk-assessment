
import pandas as pd
import numpy as np
import timeit
import sys
import os
import json
import subprocess
import logging
import pickle

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

##################Function to get model predictions
def model_predictions(model_dir, inference_df):
    #read the deployed model and a test dataset, calculate predictions
    logging.info("Calculating predictions")
    with open(os.getcwd() + '/' + model_dir + '/trainedmodel.pkl', 'rb') as file:
        model = pickle.load(file)
    
    X = inference_df.loc[:,['lastmonth_activity','lastyear_activity', 'number_of_employees']].values.reshape(-1, 3)
    predictions = model.predict(X)
    logging.info("Predictions computed")
    
    return predictions.tolist()

##################Function to get summary statistics
def dataframe_summary(data_path):
    #calculate summary statistics here
    logging.info("Calculating dataframe summary")
    df = pd.read_csv(os.getcwd() + '/' + data_path + '/finaldata.csv')
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    stats_list = []
    for col in num_cols:
        stats_list.append([df[col].mean(), df[col].median(), df[col].std()])
    logging.info("Descriptive statistics calculated")
    return stats_list

def missing_data_calculation(data_path):
    # Return the percentahe of NA per col of input dataset
    logging.info("Counting missing data")
    df = pd.read_csv(os.getcwd() + '/' + data_path + '/finaldata.csv')
    
    na_percentage_per_col = []
    for col in df.columns.values:
        na_percentage_per_col.append(df[col].isna().sum() / len(df[col]) * 100)
    
    return na_percentage_per_col
    
    
##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    logging.info("Timing ingestion and training processes")
    starttime = timeit.default_timer()
    os.system('python ingestion.py')
    timing_ingestion = timeit.default_timer() - starttime
    os.system('python3 training.py')
    timing_training = timeit.default_timer() - timing_ingestion

    return [timing_ingestion, timing_training]

##################Function to check dependencies
def outdated_packages_list():
    #get a list of dependencies used and their latest release
    logging.info("Checking dependencies")
    dep = subprocess.check_output(
        ['python',  '-m', 'pip', 'list' , '--outdated', '--format', 'columns'])
    
    return dep.decode('utf-8')
        
        
if __name__ == '__main__':
    ##################Load config.json and get environment variables
    with open('config.json','r') as f:
        config = json.load(f) 

    dataset_csv_path = os.path.join(config['output_folder_path']) 
    test_data_path = os.path.join(config['test_data_path']) 
    model_path = os.path.join(config['prod_deployment_path']) 

    inference_df = pd.read_csv(os.getcwd() + '/' + dataset_csv_path + '/finaldata.csv')
    preds = model_predictions(model_path, inference_df)
    na_info = missing_data_calculation(dataset_csv_path)
    df_sum = dataframe_summary(dataset_csv_path)
    t_ingestion, t_training = execution_time()
    deps = outdated_packages_list()

    logging.info(preds)
    logging.info(na_info)
    logging.info(df_sum)
    logging.info(t_ingestion, t_training)
    logging.info(deps)




    
