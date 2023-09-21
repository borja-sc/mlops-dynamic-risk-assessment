import pandas as pd
import numpy as np
import os
import json
from datetime import datetime


#############Function for data ingestion
def merge_multiple_dataframe(input_folder_path, output_folder_path):
    #check for datasets, compile them together, and write to an output file
    filenames = os.listdir(os.getcwd() + '/' + input_folder_path)
    merged_dataframe = pd.DataFrame()
    for f in filenames:
        currentdf = pd.read_csv(os.getcwd() + '/' + input_folder_path + '/' + f)
        merged_dataframe = merged_dataframe.append(currentdf).reset_index(drop=True)
    merged_dataframe = merged_dataframe.drop_duplicates()
    merged_dataframe.to_csv(os.getcwd() + '/' + output_folder_path + '/finaldata.csv', index=False)
    
    # Keep ingestion record
    sourcelocation = input_folder_path
    outputlocation = 'ingestedfiles.txt'
    dateTimeObj=datetime.now()
    thetimenow = str(dateTimeObj.year) + '/' + str(dateTimeObj.month) + '/' + str(dateTimeObj.day)
    allrecords=[sourcelocation, filenames, len(merged_dataframe.index), thetimenow]
    record_file = open(outputlocation,'w')
    for element in allrecords:
         record_file.write(str(element))
    record_file.close()


if __name__ == '__main__':
    #############Load config.json and get input and output paths
    with open('config.json','r') as f:
        config = json.load(f) 

    input_folder_path = config['input_folder_path']
    output_folder_path = config['output_folder_path']
    
    merge_multiple_dataframe(input_folder_path, output_folder_path)
