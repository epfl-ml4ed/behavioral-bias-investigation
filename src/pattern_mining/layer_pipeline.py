import yaml
import pickle 
import numpy as np
from typing import Tuple

class LayerPipeline:
    """This class is used to load data from algorithmic layers
    """

    def __init__(self, settings):
        self._settings = dict(settings)
        self._name = 'layer_algorithms_pipeline'
        self._notation = 'al_ppl'

    
    def _load_layers(self) -> Tuple[list, list, list]:
        """Loads the data from classification layers

        Returns:
            layers: list of weights
            demogaphics: list of dictionaries containing demographics info
            settings: updated settings if necessary
        """
        if self._settings['data']['dataset'] == 'chemlab_beerslaw':
            path = '../data/layer_weights.pkl'
            lasat_path = './configs/datasets/xxx_config.yaml'
            demographic_map = {
                'language': 'all_language',
                'labels': 'all_truths',
                'predictions': 'all_predictions',
            }
            if 'attention' in self._settings['data']['layer']:
                i_f = self._settings['data']['layer'].split('_')[-1]
                key = 'all_attention_words_five_{}'.format(i_f)
                

        with open(path, 'rb') as fp:
            layers = pickle.load(fp)

        demographics = [
            {demo: layers[demographic_map[demo]][student]
            for demo in demographic_map} 
            for student in range(len(layers[key]))
        ]

        
        with open(lasat_path, 'r') as fp:
            demos = yaml.load(fp, Loader=yaml.FullLoader)
            self._settings['pm']['demographics'] = demos['data']['demographics_maps']
            self._settings['pm']['pipeline']['demographics'] = demos['pipeline']['demographics']
        
        return layers[key], demographics, self._settings
