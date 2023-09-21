import requests
import os
import json
import sys
import logging


def main():
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)

        #Specify a URL that resolves to your workspace
        URL = "http://127.0.0.1"
        
        #Call each API endpoint and store the responses
        response1 = requests.post(URL + ':8000/prediction?data_loc=/testdata/testdata.csv').content.decode('UTF-8')
        response2 = requests.get(URL + ':8000/scoring').content.decode('UTF-8')
        response3 = requests.get(URL + ':8000/summarystats').content.decode('UTF-8')
        response4 = requests.get(URL + ':8000/diagnostics').content.decode('UTF-8')

        #combine all API responses
        responses = str(response1) + '\n\n' + str(response2) + '\n\n' + str(response3) + '\n\n' + str(response4)

        #write the responses to your workspace
        with open('config.json','r') as f:
                config = json.load(f) 

        file = open(os.getcwd() + '/' + os.path.join(config['output_folder_path']) + '/apireturns.txt','w')
        file.write(str(responses))
        file.close()

if __name__ == '__main__':
        main()