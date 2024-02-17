import os
import pickle
import logging
import itertools

import numpy as np
import pandas as pd


from ml.models.model import Model
from ml.splitters.splitter import Splitter
from ml.scorers.scorer import Scorer

class GridSearch:
    def __init__(self, model:Model, grid:dict, scorer:Scorer, splitter:Splitter, settings:dict, outer_fold:int):
        self._name = 'gridsearch'
        self._notation = 'gs'
        self._outer_fold = outer_fold
        self._model = model
        self._best_model = 'not yet'
        self._grid = dict(grid)
        self._scoring_function = scorer.get_optim_function()
        self._scoring_name = scorer.get_optim_scoring()
        self._scoring_croissant = scorer.get_optim_croissant()
        self._splitter = splitter
        self._settings = dict(settings)
        self._init_gridsearch_parameters()
        self._results = {}
        self._results_index = 0

    def get_name(self):
        return self._name
    
    def get_notation(self):
        return self._notation

    def _init_gridsearch_parameters_exhaustive(self): 
        """Initialise the combinations we will need to try
        """
        combinations = []
        parameters = []
        for param in self._grid:
            parameters.append(param)
            combinations.append(self._grid[param])
        self._combinations = list(itertools.product(*combinations))
        self._parameters = parameters

    def _init_gridsearch_parameters_combinations(self):
        """just looks into some combinations
        """
        self._parameters = self._grid['parameters']
        self._combinations = self._grid['combinations']
        
    def _init_gridsearch_parameters(self):
        if self._settings['ml']['pipeline']['parameters_gridsearch'] == 'exhaustive':
            self._init_gridsearch_parameters_exhaustive()
        elif self._settings['ml']['pipeline']['parameters_gridsearch'] == 'combinations':
            self._init_gridsearch_parameters_combinations()
        
        
    def get_parameters(self):
        return self._parameters
    
    def _add_score(self, combination:list, folds:list, fold_indices: dict):
        """Adds the scores to the list

        Args:
            combination (list): combination of parameters
            folds (list): list of all optimi_scores for each folds for that particular combination
            fold_indices (dict): dictionary with train and validation indices for each fold
        """
        score = {}
        for i, param in enumerate(self._parameters):
            score[param] = combination[i]
        score['fold_scores'] = folds
        score['mean_score'] = np.mean(folds)
        score['std_score'] = np.std(folds)
        score['fold_index'] = fold_indices
        self._results[self._results_index] = score
        self._results_index += 1
        
    def fit(self, x_train:list, y_train:list, x_test:list, y_test:list) -> dict:
        """Function to go through all parameters and find best parameters.
        All scores are computed on x_test and y_test
        Some algorithms require a validation set to avoid overfitting on the weights (particularly neural networks)
        Returns results
        """
        raise NotImplementedError
    
    def predict(self, x_test:list) -> list:
        """Predicts on the best model
        """
        raise NotImplementedError
    
    def predict_proba(self, x_test:list) -> list:
        """Predict the probabilities on the best model
        """
        raise NotImplementedError
    
    def get_best_model_settings(self) -> Model:
        """Returns the best estimator
        """
        self._results_df = pd.DataFrame.from_dict(self._results, orient='index')
        self._results_df = self._results_df.sort_values(['mean_score'], ascending=not self._scoring_croissant)
        self._best_model_settings = self._results_df.index[0]
        self._best_model_settings = self._results[self._best_model_settings]
        logging.debug('results df: {}'.format(self._results_df))
        logging.debug('best settings: {}'.format(self._best_model_settings))
        
        return self._best_model_settings
    
    def get_best_model(self) -> Model:
        return self._best_model

    def get_results(self) -> dict:
        return self._results

    def get_best_results(self) -> pd.DataFrame:
        """Returns the best results
        """
        self._results_df = pd.DataFrame.from_dict(self._results, orient='index')
        self._results_df = self._results_df.sort_values(['mean_score'], ascending=not self._scoring_croissant)
        self._best_model_settings = self._results_df.index[0]
        self._best_model_settings = self._results[self._best_model_settings]

        return self._results_df
    
    def get_path(self, fold:int) -> str:
        path = '{}/gridsearch_results/'.format(self._settings['experiment']['name'])
        os.makedirs(path, exist_ok=True)
        path += '{}/{}_l{}_f{}_modelseed{}.pkl'.format(
            self._settings['experiment']['name'],
            self._notation, str(self._settings['data']['cropper']), str(fold),
            self._settings['seeds']['model']
        )
        return path
        
    
    def save(self, fold):
        path = '{}/gridsearch_results/'.format(self._settings['experiment']['name'])
        os.makedirs(path, exist_ok=True)
        path += '{}_l{}_f{}_modelseed{}.pkl'.format(
            self._notation, str(self._settings['data']['cropper']), str(fold),
            self._settings['seeds']['model']
        )
        with open(path, 'wb') as fp:
            pickle.dump(self, fp)
        return path
    
            
        
        
        
            
            
    
            
        
