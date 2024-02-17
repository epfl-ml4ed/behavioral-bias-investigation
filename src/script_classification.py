import pickle
import yaml
import numpy as np
import argparse

from utils.config_handler import ConfigHandler
from features.pipeline_maker import PipelineMaker

from ml.xval_maker import XValMaker

def create_features(settings):
    pipeline = PipelineMaker(settings)
    states, actions, labels, demographics, indices = pipeline.load_sequences()
    print('Test processed')

def seeds_train(settings):
    handler = ConfigHandler(settings)
    settings = handler.get_experiment_name()
    feature_pipeline = PipelineMaker(settings)
    state_actions, labels, demographics, _, settings = feature_pipeline.load_sequences()
    for _ in range(settings['experiment']['model_seeds_n']):
        seed = np.random.randint(settings['experiment']['max_seed'])
        settings['seeds']['model'] = seed
        ml_pipeline = XValMaker(settings)
        ml_pipeline.train(state_actions, labels, demographics)


def test(settings):

    handler = ConfigHandler(settings)
    settings = handler.get_experiment_name()

    feature_pipeline = PipelineMaker(settings)
    state_actions, labels, demographics, indices, settings = feature_pipeline.load_sequences()

    lengths = [len(sa['state']) for sa in state_actions]
    print('Longer sequence: {}'.format(max(lengths)))
    print('done')

    # ml_pipeline = XValMaker(settings)
    # ml_pipeline.train(state_actions, labels, demographics, indices)


def main(settings):
    if settings['features']:
        create_features(settings)
    if settings['testing']:
        test(settings)
    if settings['seedstrain']:
        seeds_train(settings)

if __name__ == '__main__': 
    with open('./configs/classification_config.yaml', 'r') as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
        
    parser = argparse.ArgumentParser(description='Plot the results')
    # Tasks
    parser.add_argument('--testing', dest='testing', default=False, action='store_true')
    parser.add_argument('--features', dest='features', default=False, action='store_true')
    parser.add_argument('--seeds', dest='seedstrain', default=False, action='store_true')
    parser.add_argument('--transfer', dest='transfer', default=False, action='store_true')
    parser.add_argument('--baseline', dest='baseline', default=False, action='store_true')
    parser.add_argument('--test', dest='test', default=False, action='store_true')
    
    settings.update(vars(parser.parse_args()))
    main(settings)