import numpy as np
from features.sequencers.sequencer_tuglet import TugletSequencer

class LasatTugletSequencer(TugletSequencer):
    """
    This class prepares the DSM dataset for the tuglet dataset

    
    """

    def __init__(self, settings):
        super().__init__(dict(settings))
        self._name = 'lasta_chemlab'
        self._notation = 'las_chem'
        self._dataset = 'lasta_chem'

    def _vector_to_feature(self, timestep):
        raise NotImplementedError

    def _create_features(self, student):
        """Timesteps need to be separated by semicolons
        """
        string_student = [self._vector_to_feature(timestep) for timestep in student]
        features = '; '.join(string_student)
        return features

    def _create_sequence(self, **kwargs):
        return self._create_features(kwargs['sequence'])

                


            






