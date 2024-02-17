from argparse import ArgumentError
import numpy as np
import pandas as pd
import logging
from typing import Tuple

from sklearn.model_selection import StratifiedKFold

from ml.splitters.splitter import Splitter

class StratifiedKSplit(Splitter):
    """Stratifier that splits the data into stratified fold

    Args:
        Splitter (Splitter): Inherits from the class Splitter
    """
    
    def __init__(self, settings:dict):
        super().__init__(settings)
        self._name = 'stratified k folds'
        self._notation = 'stratkf'
        
        self._settings = dict(settings)
        self._splitter_settings = settings['ml']['splitters']
        self._random_seed = settings['seeds']['splitter']
        self._init_splitter()
        
    def set_n_folds(self, n_folds):
        if n_folds == 1:
            # n_folds = 2
            print('Should be more than 1 fold')
            raise ArgumentError
        self._n_folds = n_folds
        
    def _init_splitter(self):
        print('Splitting the data in {} folds based on seed: {}'.format(
            self._n_folds,
            self._random_seed
        ))
        self._splitter = StratifiedKFold(
            n_splits=self._n_folds,
            random_state=self._random_seed,
            shuffle=self._splitter_settings['shuffle']
        )
        
    def split(self, x:list, y:list, demographics:list) -> Tuple[list, list]:
        stratification_col = self._splitter_settings['stratifier_col']
        if stratification_col == 'y':
            return self._splitter.split(x, y)
        elif stratification_col in demographics[0]:
            demo = [student[stratification_col] for student in demographics]
            return self._splitter.split(x, demo)
        else:
            fakey = [xx[self._splitter_settings['stratifier_col']] for xx in x]
            return self._splitter.split(x, fakey)

    def next_split(self, x, y):
        return next(self.split(x, y))
            
        
        