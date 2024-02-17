import numpy as np
from sklearn.preprocessing import MinMaxScaler
class Sequencer:

    """Generate a matrix where the interaction is represented according to the sequencer's type.
    This one in particular groups all function applicatble to all sequencers

    The advantage of using a sequencer is when you need to preprocess the features differently for each dataset
    """
    def __init__(self, settings: dict):
        self._settings = dict(settings)
        self._click_length = 0.05

    def get_n_states(self):
        return self._n_states
    
    def get_n_actions(self):
        return self._n_actions

    def get_name(self):
        return self._name

    def load_sequences(self):
        raise NotImplementedError