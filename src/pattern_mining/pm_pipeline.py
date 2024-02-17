import pickle
from pattern_mining.apriori.AprioriModel import APrioriModel
from pattern_mining.lasta.lasat_model import LasatModel

class PatternMiningPipeline:
    def __init__(self, settings):
        self._settings = dict(settings)
        self._build_pipeline()

    def _choose_model(self):
        if self._settings['pm']['pipeline']['model'] == 'apriori':
            self._model = APrioriModel(self._settings)
        if self._settings['pm']['pipeline']['model'] == 'lasat':
            self._model = LasatModel(self._settings)

    def _build_pipeline(self):
        self._choose_model()

    def mine_all(self, data, demographics):
        results = self._model.mine_all(data, demographics)
        self._save(results, data, demographics)
        return results
    
    def _save(self, results, data, demographics):
        with open('{}/results.pkl'.format(self._settings['experiment']['name']), 'wb') as fp:
            pickle.dump(results, fp)

        with open('{}/data.pkl'.format(self._settings['experiment']['name']), 'wb') as fp:
            pickle.dump(data, fp)

        with open('{}/demographics.pkl'.format(self._settings['experiment']['name']), 'wb') as fp:
            pickle.dump(demographics, fp)

    