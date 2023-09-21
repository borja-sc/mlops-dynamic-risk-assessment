import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

from diagnostics import model_predictions

##############Function for reporting
def score_model(model_path, data_path, cm_save_path):
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace
    test_df = pd.read_csv(os.getcwd() + '/' + data_path + '/testdata.csv')
    y_pred = model_predictions(model_path, test_df)
    
    conf_matrix = metrics.confusion_matrix(test_df['exited'].values.reshape(-1, 1), y_pred)

    _, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    #plt.show()
    plt.savefig(os.getcwd() + '/' + cm_save_path + '/confusionmatrix.png')
    
    return



if __name__ == '__main__':
    ###############Load config.json and get path variables
    with open('config.json','r') as f:
        config = json.load(f) 

    model_path = os.path.join(config['prod_deployment_path']) 
    dataset_csv_path = os.path.join(config['test_data_path'])
    output_model_path = os.path.join(config['output_model_path'])
    
    score_model(model_path, dataset_csv_path, output_model_path)
