import json
import os
from config import config
from datetime import datetime


def save_results(results: dict):
    if not os.path.exists('results'):
        os.mkdir('results')

    dataset = config['data_dir'].split('data/')[1]

    output = {
        'config': config,
        'results': results
    }
    now = datetime.now().strftime("%Y-%m-%d %H.%M")
    path = f'results/{now} -- {dataset}.json'
    with open(path, "w") as outfile:
        outfile.write(json.dumps(output))
