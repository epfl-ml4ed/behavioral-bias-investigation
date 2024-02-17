
import os
import pickle
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import yaml
import numpy as np
import argparse

import logging
logging.INFO

from utils.config_handler import ConfigHandler
from pattern_mining.data_pipeline import PipelineMaker
from pattern_mining.layer_pipeline import LayerPipeline
from pattern_mining.pm_pipeline import PatternMiningPipeline

from ml.xval_maker import XValMaker



def create_features(settings):
    pipeline = PipelineMaker(settings)
    states, actions, labels, demographics, indices = pipeline.load_sequences()
    for student in states:
        print(student)
    print('Test processed')

def mine_patterns_layers(settings):
    handler = ConfigHandler(settings)
    settings = handler.get_patternmining_name()
    print(settings)
    logging.basicConfig(
        filename='{}/trace.log'.format(settings['experiment']['name']), encoding='utf-8', level=logging.INFO
    )

    logging.info('Starting the script...')
    layer_pipeline = LayerPipeline(settings)
    sequences, demographics, settings = layer_pipeline._load_layers()
    # print(sequences[0])
    pm_pipeline = PatternMiningPipeline(settings)
    pm_pipeline.mine_all(sequences, demographics)
    with open(settings['experiment']['name'] + 'config.pkl', 'wb') as fp:
        pickle.dump(settings, fp)

def mine_attention_layers(settings):
    for i in range(10):
        print('$' * 100)
        print('Looking into layer {}'.format(i))
        settings['experiment']['root_name'] = 'EDM24/sequences/feature_{}'.format(i)
        settings['data']['layer'] = 'attention_{}'.format(i)
        mine_patterns_layers(settings)

    

def mine_patterns_features(settings):
    """ Mining the input sequences given to the success predictors """
    handler = ConfigHandler(settings)
    settings = handler.get_patternmining_name()
    logging.basicConfig(
        filename='{}/trace.log'.format(settings['experiment']['name']), encoding='utf-8', level=logging.INFO
    )
    logging.info('Starting the script...')
    feature_pipeline = PipelineMaker(settings)
    sequences, _, demographics, _, settings = feature_pipeline.load_sequences()
    pm_pipeline = PatternMiningPipeline(settings)
    pm_pipeline.mine_all(sequences, demographics)
    with open(settings['experiment']['name'] + 'config.pkl', 'wb') as fp:
        pickle.dump(settings, fp)


def test(settings):

    handler = ConfigHandler(settings)
    settings = handler.get_patternmining_name()
    print(settings)
    feature_pipeline = PipelineMaker(settings)
    state_actions, labels, demographics, indices, settings = feature_pipeline.load_sequences()

    lengths = [len(sa['state']) for sa in state_actions]
    print('Longer sequence: {}'.format(max(lengths)))
    print('done')

    # ml_pipeline = XValMaker(settings)
    # ml_pipeline.train(state_actions, labels, demographics, indices)


def main(settings):
    if settings['mine']:
        if settings['sequences']:
            mine_patterns_features(settings)
        if settings['layers']:
            mine_patterns_layers(settings)
        if settings['attention']:
            mine_attention_layers(settings)
    if settings['features']:
        create_features(settings)
    if settings['test']:
        test(settings)

if __name__ == '__main__': 
    with open('./configs/pattern_config.yaml', 'r') as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
        
    parser = argparse.ArgumentParser(description='Plot the results')
    # Tasks
    parser.add_argument('--mine', dest='mine', default=False, action='store_true')
    parser.add_argument('--sequences', dest='sequences', default=False, action='store_true')
    parser.add_argument('--layers', dest='layers', default=False, action='store_true')
    parser.add_argument('--attention', dest='attention', default=False, action='store_true')
    parser.add_argument('--features', dest='features', default=False, action='store_true')
    parser.add_argument('--test', dest='test', default=False, action='store_true')
    
    settings.update(vars(parser.parse_args()))
    main(settings)