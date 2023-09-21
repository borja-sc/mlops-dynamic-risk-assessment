
import os
import json
import logging
import sys

import ingestion
import training
import scoring
import deployment

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

with open('config.json','r') as f:
        config = json.load(f) 
INPUT_FOLDER_PATH = config["input_folder_path"]
OUTPUT_FOLDER_PATH = config["output_folder_path"]
PROD_DEPLOYMENT_PATH = config["prod_deployment_path"]
TEST_DATA_PATH = config["test_data_path"]
OUTPUT_MODEL_PATH = config["output_model_path"]

def main():
    ##################Check and read new data
    # Read ingestedfiles.txt
    logging.info("Checking for new data")
    f = open(PROD_DEPLOYMENT_PATH + "/ingestedfiles.txt", "r")
    ingested = f.readline()
    f.close()

    #second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
    source_files = set(os.listdir(INPUT_FOLDER_PATH))
    deploy = False
    for f in source_files:
        if f not in ingested:
            deploy = True
            ingestion.merge_multiple_dataframe(INPUT_FOLDER_PATH, OUTPUT_FOLDER_PATH)

    ##################Deciding whether to proceed, part 1
    #if you found new data, you should proceed. otherwise, do end the process here
    if deploy == False:
        logging.info("No new data. No re-deployment will take place.")
        return None
    logging.info("New data detected. Comparing F1 scores.")

    ##################Checking for model drift
    #check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
    with open(PROD_DEPLOYMENT_PATH + "/latestscore.txt", "r") as f:
        deployed_score = float(f.readline())

    new_score = scoring.score_model(PROD_DEPLOYMENT_PATH, TEST_DATA_PATH)
    ##################Deciding whether to proceed, part 2
    #if you found model drift, you should proceed. otherwise, do end the process here
    if new_score > deployed_score:
        logging.info("New F1 score higher than thedeployed. No re-deployment will take place.")
        return None
    logging.info("New F1 score lower than the deployed. Starting re-deployment.")

    ##################Re-deployment
    #if you found evidence for model drift, re-run the deployment.py script
    logging.info("Re-training")
    training.train_model(OUTPUT_FOLDER_PATH, OUTPUT_MODEL_PATH)
    logging.info("Re-deployment")
    deployment.store_model_into_pickle(OUTPUT_MODEL_PATH, PROD_DEPLOYMENT_PATH)

    ##################Diagnostics and reporting
    #run apicalls.py and reporting.py for the re-deployed model
    logging.info("Running diagnostics for new model")
    os.system("python apicalls.py")
    os.system("python reporting.py")
    


if __name__ == '__main__':
    main()



