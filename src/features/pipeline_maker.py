
from features.sequencers.sequencer import Sequencer # Look into sequencer to create your own, or directly load it here
class PipelineMaker:

    def __init__(self, settings):
        self._settings = dict(settings)

    def _select_sequencer(self):
        if self._settings['data']['dataset'] == 'datasetname': # argument in classification config
            self._settings['ml']['models']['maxlen'] = 1000 # longest sequence in the
            if self._settings['data']['sequencer'] == 'generalised': # argument in classification config
                self._sequencer = Sequencer(self._settings)
                self._settings['ml']['models']['maxlen'] = 942


    def load_sequences(self):
        """Loads the sequences based on the classification config
        Returns:
            sequences: sequences of interactions (list of all students' interactions)
            labels: classification labels
            demographics: (list of dictionaries, one for each student
        where the key is the name of the demographic attribute)
            settings: the settings of the whole experiment

        """
        self._select_sequencer()

        sequences, labels, demographics, indices = self._sequencer.load_all_sequences()
        return sequences, labels, demographics, indices, self._settings

