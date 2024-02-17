import os
import logging
import pickle
import numpy as np
import pandas as pd
from typing import Tuple
from shutil import copytree, rmtree 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from ml.models.model import Model
from tensorflow.keras import Model as Mod

import tensorflow as tf
# tf.get_logger().setLevel('INFO')
# tf.get_logger().setLevel(logging.ERROR)

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import History
from tensorflow.keras.losses import get as get_loss, Loss
from sklearn.model_selection import KFold, StratifiedKFold
from tensorflow.keras.metrics import get as get_metric, Metric
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from silence_tensorflow import silence_tensorflow
silence_tensorflow()


from numpy.random import seed

class LSTMKerasModel(Model):
    """This class implements an LSTM
    Args:
        Model (Model): inherits from the model class
    """
    
    def __init__(self, settings:dict):
        super().__init__(settings)
        self._name = 'long short term memory'
        self._notation = 'lstm'
        self._model_settings = settings['ml']['models']['lstm']
        self._maxlen = self._settings['ml']['models']['maxlen']
        self._padding_value = self._model_settings['padding_value']
        self._fold = 0


    def _format(self, x:list, y:list) -> Tuple[list, list]:
        #y needs to be one hot encoded
        x_vector = pad_sequences(x, padding="post", value=self._padding_value, maxlen=self._maxlen, dtype=float)
        print('input format: ', x_vector.shape)
        y_vector = to_categorical(y, num_classes=self._n_classes)
        return x_vector, y_vector
    
    def _format_features(self, x:list) -> list:
        x_vector = pad_sequences(x, padding="post", value=self._padding_value, maxlen=self._maxlen, dtype=float)
        return x_vector
    
    def _get_rnn_layer(self, return_sequences:bool, l:int):
        n_cells = self._model_settings['n_cells'][l]
        if self._model_settings['cell_type'] == 'LSTM':
            layer = layers.LSTM(units=n_cells, return_sequences=return_sequences)
        elif self._model_settings['cell_type'] == 'GRU':
            layer = layers.GRU(units=n_cells, return_sequences=return_sequences)
        elif self._model_settings['cell_type'] == 'RNN':
            layer = layers.SimpleRNN(units=n_cells, return_sequences=return_sequences)
        elif self._model_settings['cell_type'] == 'BiLSTM':
            layer = layers.LSTM(units=n_cells, return_sequences=return_sequences)
            layer = layers.Bidirectional(layer=layer)
        return layer

    def _get_csvlogger_path(self) -> str:
        csv_path = '../experiments/{}/{}/logger_{}/ct{}_nlayers{}_ncells{}_drop{}_optim{}_loss{}_bs{}_ep{}_seed{}'.format(
            self._experiment_name, self._outer_fold, self._notation,
            self._model_settings['cell_type'], self._model_settings['n_layers'], self._model_settings['n_cells'],
            str(self._model_settings['dropout']).replace('.', ''), self._model_settings['optimiser'], self._model_settings['loss'],
            self._model_settings['batch_size'], self._model_settings['epochs'], self._settings['seeds']['model']
        )
        if self._gs_fold != -1:
            csv_path += '/f{}_model_'.format(str(self._gs_fold))

        os.makedirs(csv_path, exist_ok=True)
        checkpoint_path = csv_path + 'checkpoint/'
        csv_path +=  'training.csv'
        return csv_path, checkpoint_path

    def load_model_weights(self, x:np.array, checkpoint_path:str):
        """Given a data point x, this function sets the model of this object
        Args:
            x ([type]): [description]
        Raises:
            NotImplementedError: [description]
        """
        # x = self._format_features(x) 
        self._init_model(x)
        cce = tf.keras.losses.CategoricalCrossentropy(name='categorical_crossentropy')
        auc = tf.keras.metrics.AUC(name='auc')
        self._model.compile(
            loss=['categorical_crossentropy'], optimizer='adam', metrics=[cce, auc]
        )
        checkpoint = tf.train.Checkpoint(self._model)
        temporary_path = '../experiments/temp_checkpoints/training/'
        if os.path.exists(temporary_path):
            rmtree(temporary_path)
        copytree(checkpoint_path, temporary_path, dirs_exist_ok=True)
        checkpoint.restore(temporary_path).expect_partial()

    def _init_model(self, x:np.array):
        # initial layers
        self._set_seed()

        input_feature = layers.Input(shape=(x.shape[1], x.shape[2]), name='input_features')
        full_features = layers.Masking(mask_value=self._model_settings['padding_value'], name='masking_features')(input_feature)

        for l in range(int(self._model_settings['n_layers']) -1):
            full_features = self._get_rnn_layer(return_sequences=True, l=l)(full_features)
        full_features = self._get_rnn_layer(return_sequences=False, l=self._model_settings['n_layers'] - 1)(full_features)

        if self._model_settings['dropout'] != 0.0:
            full_features = layers.Dropout(self._model_settings['dropout'])(full_features)

        classification_layer = layers.Dense(self._settings['experiment']['nclasses'], activation='softmax')(full_features)

        self._model = Mod(input_feature, classification_layer)


        # compiling
        cce = tf.keras.losses.CategoricalCrossentropy(name='categorical_crossentropy')
        auc = tf.keras.metrics.AUC(name='auc')
        self._model.compile(
            loss=['categorical_crossentropy'], optimizer='adam', metrics=[cce, auc]
        )
        
        # callbacks
        self._callbacks = []
        if self._model_settings['early_stopping']:
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, min_delta=0.001, 
                restore_best_weights=True
            )
            self._callbacks.append(early_stopping)
            
        # csv loggers
        csv_path, checkpoint_path = self._get_csvlogger_path()
        csv_logger = CSVLogger(csv_path, append=True, separator=';')
        self._callbacks.append(csv_logger)

        if self._model_settings['save_best_model']:
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_auc',
            mode='max',
            save_best_only=True)
            self._callbacks.append(model_checkpoint_callback)

        print(self._model.summary())

    def fit(self, x_train:list, y_train:list, x_val:list, y_val:list):
        x_train, y_train = self._format(x_train, y_train)
        x_val, y_val = self._format(x_val, y_val)

        self._init_model(x_train)
        self._history = self._model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            batch_size=self._model_settings['batch_size'],
            shuffle=self._model_settings['shuffle'],
            epochs=self._model_settings['epochs'],
            verbose=self._model_settings['verbose'],
            callbacks=self._callbacks
        )

        if self._model_settings['save_best_model']:
            checkpoint_path = self._get_model_checkpoint_path()
            self.load_model_weights(x_train, checkpoint_path)
            self._best_epochs = np.argmax(self._history.history['val_auc'])
        self._fold += 1
        
    def predict(self, x:list) -> list:
        return self.predict_tensorflow(x)
    
    def predict_proba(self, x:list) -> list:
        probs = self.predict_proba_tensorflow(x)
        return probs
    
    def save(self) -> str:
        self.save_tensorflow()
    
    def get_path(self, fold: int) -> str:
        self.get_path(fold)
            
    def save_fold(self, fold: int) -> str:
        self.save_fold_tensorflow(fold)

    def save_fold_early(self, fold: int) -> str:
        return self.save_fold_early_tensorflow(fold)