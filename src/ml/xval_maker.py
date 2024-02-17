import yaml
import logging
from ml.models.lstm_keras import LSTMKerasModel

from ml.models.rnn_attention import RNNAttentionModel

from ml.samplers.no_sampler import NoSampler

from ml.gridsearches.supervised_gridsearch import SupervisedGridSearch

from ml.scorers.binaryclassification_scorer import BinaryClfScorer
from ml.scorers.multiclassification_scorer import MultiClfScorer

from ml.splitters.splitter import Splitter
from ml.splitters.stratified_kfold import StratifiedKSplit

from ml.xvalidators.nonnested_xval import NonNestedRankingXVal

class XValMaker:
    """This script assembles the machine learning component and creates the training pipeline according to:
    
        - splitter
        - sampler
        - model
        - xvalidator
        - scorer
    """
    
    def __init__(self, settings:dict):
        logging.debug('initialising the xval')
        self._name = 'training maker'
        self._notation = 'trnmkr'
        self._settings = dict(settings)
        self._experiment_root = self._settings['experiment']['root_name']
        self._experiment_name = settings['experiment']['name']
        self._pipeline_settings = self._settings['ml']['pipeline']
        
        self._build_pipeline()
        

    def get_gridsearch_splitter(self):
        return self._gs_splitter

    def get_sampler(self):
        return self._sampler

    def get_scorer(self):
        return self._scorer

    def get_model(self):
        return self._model

    def _choose_splitter(self, splitter:str) -> Splitter:
        if splitter == 'stratkf':
            return StratifiedKSplit
    
    def _choose_inner_splitter(self): # only for nested xval
        self._inner_splitter = self._choose_splitter(self._pipeline_settings['inner_splitter'])

    def _choose_outer_splitter(self):
        self._outer_splitter = self._choose_splitter(self._pipeline_settings['outer_splitter'])

    def _choose_gridsearch_splitter(self):
        self._gs_splitter = self._choose_splitter(self._pipeline_settings['gs_splitter'])
            
    def _choose_sampler(self):
        if self._pipeline_settings['sampler'] == 'nosplr':
            self._sampler = NoSampler
            
    def _choose_model(self):
        logging.debug('model: {}'.format(self._pipeline_settings['model']))
        if self._pipeline_settings['model'] == 'lstm_keras':
            self._model = LSTMKerasModel
            gs_path = './configs/gridsearch/gs_lstm.yaml'


        if self._pipeline_settings['model'] == 'rnn_attention':
            self._model = RNNAttentionModel
            gs_path = './configs/gridsearch/gs_lstm.yaml' # TO CHANGE
            
        if self._settings['ml']['pipeline']['gridsearch'] != 'nogs':
            with open(gs_path, 'r') as fp:
                gs = yaml.load(fp, Loader=yaml.FullLoader)
                self._settings['ml']['xvalidators']['nested_xval']['paramgrid'] = gs
                print(gs)
                    
    def _choose_scorer(self):
        if self._pipeline_settings['scorer'] == '2clfscorer':
            self._scorer = BinaryClfScorer
        elif self._pipeline_settings['scorer'] == 'multiclfscorer':
            self._scorer = MultiClfScorer

    def _choose_gridsearcher(self):
        if self._pipeline_settings['gridsearch'] == 'supgs':
            self._gridsearch = SupervisedGridSearch
                
    def _choose_xvalidator(self):
        if 'nested' in self._pipeline_settings['xvalidator']:
            self._choose_gridsearcher()
        if self._pipeline_settings['xvalidator'] == 'nonnested_xval':
            self._xval = NonNestedRankingXVal

        self._xval = self._xval(self._settings, self._gridsearch, self._gs_splitter, self._outer_splitter, self._sampler, self._model, self._scorer)
    
    def _train_non_gen(self, X:list, y:list, demographics:list):
        results = self._xval.xval(X, y, demographics)
        return results

    def _choose_train(self):
        self.train = self._train_non_gen

    def _build_pipeline(self):
        # self._choose_splitter()
        # self._choose_inner_splitter()
        self._choose_outer_splitter()
        self._choose_gridsearch_splitter()
        self._choose_sampler()
        self._choose_model()
        self._choose_scorer()
        self._choose_xvalidator()
        self._choose_train()
        
    
        