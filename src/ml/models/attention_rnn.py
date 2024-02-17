import os
import logging
import pickle
from unicodedata import bidirectional
from xml.sax.xmlreader import AttributesNSImpl
import numpy as np
import pandas as pd
from typing import Tuple
from shutil import copytree, rmtree
from ml.models.model import Model

from numpy.random import seed
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.masked import masked_tensor, as_masked_tensor

class AttentionRNN(nn.Module):
    def __init__(self, settings):
        super(AttentionRNN, self).__init__()
        self._model_settings = dict(settings['ml']['models']['attention_rnn'])
        print('ms', self._model_settings)
        self.device = 'cpu'
        self._n_classes = settings['n_classes']

        # attention layers
        self.attention_hidden_size = self._model_settings['attention_hidden_size']
        self.key = torch.nn.Linear(self._model_settings['input_dimension'][-1], self.attention_hidden_size)
        self.value = torch.nn.Linear(self._model_settings['input_dimension'][-1], self.attention_hidden_size)
        self.query = torch.nn.Linear(self._model_settings['input_dimension'][-1], self.attention_hidden_size)
        self.softmax_attention = torch.nn.Softmax(dim=-1)

        # Gru layers
        self.gru_hidden_size = self._model_settings['rnn_ncells']
        self.num_directions = 2 if 'bi' in self._model_settings['rnn_cell_type'] == True else 1
        if self._model_settings['attention_agg'] == 'concat':
            gru_size = self.attention_hidden_size + self._model_settings['input_dimension'][-1]
        else:
            gru_size = self.attention_hidden_size
        self.gru = torch.nn.GRU(
            gru_size,
            self.gru_hidden_size,
            num_layers=self._model_settings['rnn_nlayers'],
            dropout=self._model_settings['rnn_dropout'],
            bidirectional=bool(self.num_directions-1),
            batch_first=True
        )
        self.hidden = None

        # Dense layers
        self.dropout = nn.Dropout(p=self._model_settings['classifier_dropout'])
        self.linear = nn.Linear(self.gru_hidden_size, self._n_classes)
        # torch.nn.init.xavier_uniform_(self.linear.weight)
        # self.linear.bias.data.fill_(0.0)

    def get_attention_weights(self, sequences):
        _, aw = self._attention(sequences)
        return aw

    def _concat_attention_sequences(self, attention_outputs, sequences):
        if self._model_settings['attention_agg'] == 'none':
            return attention_outputs

        elif self._model_settings['attention_agg'] == 'concat':
            return torch.cat([attention_outputs, sequences], axis=2)

        elif self._model_settings['attention_agg'] == 'addition':
            return attention_outputs + sequences

        elif self._model_settings['attention_agg'] == 'multiplication':
            return attention_outputs * sequences

    def _attention(self, sequences):
        if self._model_settings['attention_type'] == 'kqv':
            queries = self.query(sequences)
            keys = self.key(sequences)
            values = self.value(sequences)
            scores = torch.bmm(queries, keys.transpose(1, 2)) / (10 ** 0.5)
            self._attention_weights = self.softmax_attention(scores)
            attention_outputs = torch.bmm(self._attention_weights, values)
            concat_attention_outputs = self._concat_attention_sequences(attention_outputs, sequences)
            return concat_attention_outputs, self._attention_weights

        elif self._model_settings['attention_type'] == 'keras':
            m = torch.matmul(sequences, sequences.transpose(-2, -1))
            self._attention_weights = F.softmax(m, dim=-1)
            o = torch.matmul(self._attention_weights, sequences)
            a = torch.mul(o, sequences)
            concat_attention_outputs = self._concat_attention_sequences(a, sequences)
            return concat_attention_outputs, self._attention_weights

    def init_hidden(self, batch_size):
        return torch.nn.init.orthogonal_(torch.zeros(
            self._model_settings['rnn_nlayers'] * self.num_directions,
            batch_size,
            self.gru_hidden_size
        ))

    def _gru(self, attention_outputs, lengths):
        # Keras
        if 'keras' in self._model_settings['rnn_type']:
            rnn_x, _ = self.gru(attention_outputs, self.hidden)
            rnn_x = rnn_x.detach().numpy()
            if 'accumul' in self._model_settings['rnn_type']:
                rnn_x = [np.mean(rnn_x[i][:int(lengths[i])-1], axis=0) for i in range(len(rnn_x))]
            else:
                rnn_x = [rnn_x[i][int(lengths[i])-1] for i in range(len(rnn_x))]
            rnn_x = torch.Tensor(rnn_x)
            return rnn_x

        # agg
        if self._model_settings['rnn_type'] == 'agg':
            rnn_x, _ = self.gru(attention_outputs, self.hidden)
            return torch.mean(rnn_x, 1)

        # pad
        if self._model_settings['rnn_type'] == 'pad':
            attention_outputs = torch.nn.utils.rnn.pack_padded_sequence(attention_outputs, lengths, batch_first=True, enforce_sorted=False)
            rnn_x, hn = self.gru(attention_outputs, self.hidden)
            rnn_x, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_x, batch_first=True)
            return hn[0]

    def _classifier(self, rnn_last_final):
        # clf_outputs = self.dropout(rnn_last_final)
        # clf_outputs = self.linear(clf_outputs)
        clf_outputs = self.linear(rnn_last_final)
        clf_outputs = F.log_softmax(clf_outputs, dim=1)
        return clf_outputs

    def forward(self, sequences, lens):
        attention_outputs, attention_weights = self._attention(sequences)
        final_rnn_state = self._gru(attention_outputs, lens)
        # print('final', final_rnn_state)
        output = self._classifier(final_rnn_state)
        return output, [attention_outputs, attention_weights, final_rnn_state]



class AttentionRNNModel(Model):
    """This class implements an LSTM with attention where the last timestamp of the lstm layer is concatenated with the attention output
    Args:
        Model (Model): inherits from the model class


    Implemented with the help of this repository:
        https://github.com/chrisvdweth/ml-toolkit/blob/master/pytorch/models/text/classifier/rnn.py and
    """
    
    def __init__(self, settings:dict):
        super().__init__(settings)
        self._name = 'last timestep attention pytorch'
        self._notation = 'ltsattpt'
        self._model_settings = dict(settings['ml']['models']['attention_rnn'])
        self._maxlen = self._settings['ml']['models']['maxlen']
        self._fold = 0

    def _format(self, x:list, y:list) -> Tuple[list, list]:
        x_tensor, x_lens = self._format_features(x)
        y_tensor = torch.LongTensor(y)
        return x_tensor, x_lens, y_tensor
    
    def _format_features(self, x:list) -> list:
        xlens = torch.Tensor([len(student) for student in x])
        padded_sequence = pad_sequences(x, padding='post', value=self._model_settings['padding_value'], maxlen=self._maxlen, dtype=float)
        x_tensor = torch.Tensor(padded_sequence)
        return x_tensor, xlens


    def _init_model(self, x=''):
        if x != '':
            self._settings['ml']['models']['attention_rnn']['input_dimension'] = x.size()
        self._settings['ml']['models']['attention_rnn'].update(self._model_settings)
        self.model = AttentionRNN(self._settings)
        if self._model_settings['loss_name'] == 'nll':
            self.criterion = torch.nn.NLLLoss(reduction=self._model_settings['loss_reduction'])
        elif self._model_settings['loss_name'] == 'cce':
            self.criterion = torch.nn.CrossEntropyLoss(reduction=self._model_settings['loss_reduction'])

        self.optimiser = optim.Adam(self.model.parameters(), eps=1e-07)

    def fit(self, x_train:list, y_train:list, x_val:list, y_val:list):
        # initialise data
        x_train, x_lens, y_train = self._format(x_train, y_train)
        x_val, val_lens, y_val = self._format(x_val, y_val)
        dataset = TensorDataset(x_train, x_lens, y_train)
        dataloader = DataLoader(dataset, batch_size=self._model_settings['batch_size'], shuffle=True)
        self._settings['ml']['models']['attention_rnn']['input_dimension'] = x_train.size()
        self._settings['n_classes'] = self._n_classes

        # initialise model
        self._init_model()
        for epoch in range(self._model_settings['epochs']):
            print('epoch {}'.format(epoch))
            self.model.train()
            losses = 0
            for batch_x, batch_lens, batch_ys in dataloader:
                self.optimiser.zero_grad()
                self.model.hidden = self.model.init_hidden(batch_x.size(0))
                log_probs, _ = self.model(batch_x, batch_lens)
                loss = self.criterion(log_probs, batch_ys)
                losses += loss
                loss.backward()
                # self.getBack(loss.grad_fn)
                self.optimiser.step()
            # print('Losses: ', losses)

    def predict(self, x:list) -> list:
        x_test, x_lens = self._format_features(x)
        dataset = TensorDataset(x_test, x_lens)
        testloader = DataLoader(dataset, batch_size=self._model_settings['batch_size'], shuffle=False)
        
        self.model.eval()
        predictions = torch.Tensor()
        with torch.no_grad():
            for batch_x, batch_len in testloader:
                self.model.hidden = self.model.init_hidden(batch_x.size(0))
                log_probs, _ = self.model(batch_x, batch_len)
                print('log', log_probs.size(), batch_x.size())
                _, preds = torch.max(log_probs, 1)
                predictions = torch.cat((predictions, preds))
        print(predictions.size())
        print(predictions)
        return predictions
    
    def predict_proba(self, x:list) -> list:
        x_test, x_lens = self._format_features(x)
        dataset = TensorDataset(x_test, x_lens)
        testloader = DataLoader(dataset, batch_size=self._model_settings['batch_size'], shuffle=False)
        
        self.model.eval()
        predictions = torch.Tensor()
        with torch.no_grad():
            for batch_x, batch_len in testloader:
                self.model.hidden = self.model.init_hidden(batch_x.size(0))
                log_probs, _ = self.model(batch_x, batch_len)
                probs = F.softmax(log_probs)
                predictions = torch.cat((predictions, probs))

        print('probas  ')
        print(predictions.size())
        print(predictions)
        return predictions

    def get_gru_outputs(self, x: list) -> list:
        x, x_lens = self._format_features(x)
        dataset = TensorDataset(x, x_lens)
        testloader = DataLoader(dataset, batch_size=self._model_settings['batch_size'], shuffle=False)
        
        self.model.eval()
        gru_weights = torch.Tensor()
        with torch.no_grad():
            for batch_x, batch_len in testloader:
                self.model.hidden = self.model.init_hidden(batch_x.size(0))
                _, weights = self.model(batch_x, batch_len)
                gru_weights = torch.cat((gru_weights, weights[2]))
        return gru_weights

    def get_attention_outputs(self, x: list) -> list:
        x, x_lens = self._format_features(x)
        dataset = TensorDataset(x, x_lens)
        testloader = DataLoader(dataset, batch_size=self._model_settings['batch_size'], shuffle=False)
        
        self.model.eval()
        attention_weights = torch.Tensor()
        with torch.no_grad():
            for batch_x, batch_len in testloader:
                self.model.hidden = self.model.init_hidden(batch_x.size(0))
                _, weights = self.model(batch_x, batch_len)
                attention_weights = torch.cat((attention_weights, weights[0]))
        return attention_weights

    def _get_model_checkpoint_path(self) -> str:
        path = '{}{}/logger/{}/'.format(self._experiment_name, self._outer_fold, self._notation)
        path += 'ct{}_nlayers{}_ncells{}'.format(
            self._model_settings['rnn_cell_type'], 
            self._model_settings['rnn_nlayers'], 
            self._model_settings['rnn_ncells']
        )
        path += '_bs{}_ep{}/'.format(
            self._model_settings['batch_size'], self._model_settings['epochs']
        )
        # path += '/f{}_model_checkpoint/'.format(self._gs_fold)
        return path

    def load_model_weights(self, x: np.array, checkpoint_path: str):
        x, _ = self._format_features(x)
        self._init_model(x)
        # print(list(self.model.gru.parameters())[0])
        model_pth = torch.load(checkpoint_path)
        self.model.load_state_dict(model_pth)
        # print(list(self.model.gru.parameters())[0])
        self.model.eval()
    
    def save(self, extension='') -> str:
        return self.save_pytorch(extension)

    def save_fold(self, fold: int) -> str:
        return self.save()
    