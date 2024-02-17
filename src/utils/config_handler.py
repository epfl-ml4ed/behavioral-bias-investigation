import os
from os import path as pth
from datetime import datetime
import pickle

class ConfigHandler:
    def __init__(self, settings:dict):
        self._settings = settings
        
    def get_settings(self):
        return dict(self._settings)

    def get_experiment_name(self):
        """Creates the experiment name in the following path:
            '../experiments/experiment root/yyyy_mm_dd_index/'
            index being the first index in increasing order starting from 0 that does not exist yet.
            
            This function:
            - returns the experiment config name 
            - creates the folder with the right experiment name at ../experiments/experiment root/yyyy_mm_dd_index
            - dumps the config in the newly created folder

        Args:
            settings ([type]): read config

        Returns:
            [str]: Returns the name of the experiment in the format of 'yyyy_mm_dd_index'
        """

        path = '/data{}/m{}_f{}/'.format(
            self._settings['data']['dataset'].replace('.', '-'),
            self._settings['ml']['pipeline']['model'], self._settings['ml']['splitters']['nfolds']
        )
        today = datetime.today().strftime('%Y-%m-%d')
        today = today.replace('-', '_')
        starting_index = 0
        
        # first index
        experiment_name = '../experiments/{}{}{}_{}/'.format(
            self._settings['experiment']['root_name'], path, today, starting_index
        )
        while (pth.exists(experiment_name)):
            starting_index += 1
            experiment_name = '../experiments/{}{}{}_{}/'.format(
                self._settings['experiment']['root_name'], path, today, starting_index
            )
            
        self._experiment_path = experiment_name
        os.makedirs(self._experiment_path, exist_ok=True)
        self._settings['experiment']['name'] = self._experiment_path

        with open(self._settings['experiment']['name'] + 'config.pkl', 'wb') as fp:
            pickle.dump(self._settings, fp)

        return self._settings

    ##################### Pattern Mining #####################

    def get_patternmining_name(self):
        """Creates the experiment name in the following path:
            '../experiments/experiment root/yyyy_mm_dd_index/'
            index being the first index in increasing order starting from 0 that does not exist yet.
            
            This function:
            - returns the experiment config name 
            - creates the folder with the right experiment name at ../experiments/experiment root/yyyy_mm_dd_index
            - dumps the config in the newly created folder

        Args:
            settings ([type]): read config

        Returns:
            [str]: Returns the name of the experiment in the format of 'yyyy_mm_dd_index'
        """

        demographic_pipe = [demo_group.replace('.', '*').replace(',', '*') for demo_group in self._settings['pm']['pipeline']['demographics']]
        demographic_pipe = [dg.split('*') for dg in demographic_pipe]
        demographic_pipe = [[ddg[:3] for ddg in dg] for dg in demographic_pipe]
        demographic_pipe = ['-'.join(dg) for dg in demographic_pipe]
        demographic_pipe = '_'.join(demographic_pipe)
        path = '/data{}/m{}/demographics_{}/'.format(
            self._settings['data']['dataset'].replace('.', '-'),
            self._settings['pm']['pipeline']['model'], demographic_pipe
        )
        today = datetime.today().strftime('%Y-%m-%d')
        today = today.replace('-', '_')
        starting_index = 0
        
        # first index
        experiment_name = '../experiments/pattern-mining/{}{}{}_{}/'.format(
            self._settings['experiment']['root_name'], path, today, starting_index
        )
        while (pth.exists(experiment_name)):
            starting_index += 1
            experiment_name = '../experiments/pattern-mining/{}{}{}_{}/'.format(
                self._settings['experiment']['root_name'], path, today, starting_index
            )
            
        self._experiment_path = experiment_name
        os.makedirs(self._experiment_path, exist_ok=True)
        self._settings['experiment']['name'] = self._experiment_path

        with open(self._settings['experiment']['name'] + 'config.pkl', 'wb') as fp:
            pickle.dump(self._settings, fp)

        return self._settings
      