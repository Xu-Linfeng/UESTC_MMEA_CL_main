import json
import argparse
from trainer import train
from opts import parser
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

def main():
    args = parser.parse_args()
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json
    
    train(args)
    

def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param


if __name__ == '__main__':
    main()
