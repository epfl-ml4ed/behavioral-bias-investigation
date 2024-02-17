import yaml
import logging

from pattern_mining.features.beerslaw_apriori import AprioriChemLabSequencer
from pattern_mining.features.beerslaw_lasat import LasatChemLabSequencer
from pattern_mining.features.tuglet_apriori import AprioriTugletSequencer
from pattern_mining.features.tuglet_lasat import LasatTugletSequencer
from pattern_mining.features.tuglet_dmkt import DMKTTugletSequencer

class PipelineMaker:

    def __init__(self, settings):
        self._settings = dict(settings)

    def _select_sequencer(self):
        if self._settings['data']['dataset'] == 'tuglet':
            path = './configs/datasets/xxx.yaml'
            if self._settings['pm']['pipeline']['model'] == 'example':
                self._sequencer =  LasatTugletSequencer(self._settings)

        with open(path, 'r') as fp:
            demos = yaml.load(fp, Loader=yaml.FullLoader)
            self._settings['pm']['demographics'] = demos['data']['demographics_maps']
            self._settings['pm']['pipeline']['demographics'] = demos['pipeline']['demographics']
            
    def load_sequences(self):
        self._select_sequencer()
        state_actions, labels, demographics, indices = self._sequencer.load_all_sequences()
        logging.info('Sequences loaded using the {} sequencer...'.format(self._sequencer.get_name()))
        return state_actions, labels, demographics, indices, self._settings

