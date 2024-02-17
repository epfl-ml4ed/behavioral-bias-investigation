import os
import pickle

import numpy as np
import pandas as pd

import torch
from clustering.clustering_pipeline import ClusteringPipeline

import seaborn as sns
from matplotlib import pyplot as plt
from visualisers.stylers.full_sequences_styler import FullStyler

class ModelPlotter:
    def __init__(self, settings):
        self._name = 'model_plotter'
        self._settings = settings
        self._styler = FullStyler(settings)
        self._crawled = False

    def _crawl(self):
        if self._crawled:
            return {}
        else:
            self._crawled = True
            # crawl paths
            model_paths = []
            experiment_path = '../experiments/' + self._settings['experiment']['name'] + '/'
            for (dirpath, _, filenames) in os.walk(experiment_path):
                files = [os.path.join(dirpath, file) for file in filenames]
                model_paths.extend(files)

            torch_models = [model for model in model_paths if 'torch_object.pt' in model]
            models = {}
            for torch_path in torch_models:
                pickle_path_1 = torch_path.replace('_torch_object.pt', 'model.pkl')
                pickle_path_2 = torch_path.replace('_torch_object', 'model.pkl')
                if pickle_path_1 in model_paths:
                    with open(pickle_path_1, 'rb') as fp:
                        p_model = pickle.load(fp)
                    t_model = torch.load(torch_path)
                elif pickle_path_2 in model_paths:
                    with open(pickle_path_2, 'rb') as fp:
                        p_model = pickle.load(fp)
                    t_model = torch.load(torch_path)
                else:
                    print(torch_path)
                    with open(torch_path, 'rb') as fp:
                        p_model = pickle.load(fp)
                    t_model = p_model.get_model()

                key_path = torch_path.replace('_torch_object.pt', '')
                models[key_path] = {
                    'torch': t_model,
                    'pickle': p_model
                }

            print('Here are all the models we found:')
            model_keys = []
            for i_m, model_path in enumerate(models):
                print('{}: {}'.format(i_m, model_path))
                model_keys.append(model_path)
            
            model_index = input('Which one would you like to do the visualisation for ? (Enter Integer number. Do not try to trick the system, you will manage, there will be a crash.)\n')
            model_index = int(model_index)
            self._settings['results_path'] = '{}/overall_results/'.format(model_keys[model_index])
            os.makedirs(self._settings['results_path'], exist_ok=True)
            self._models = models[model_keys[model_index]]

    def get_correlation_weights(self, x):
        self._crawl()
        unique_questions = self._get_feature_activations(x)
        question_weights = {}
        for question in unique_questions:
            w = self._models['torch'].get_weight_correlation(question)
            wcpu = w.cpu().data.numpy()
            question_weights['_'.join([str(qq) for qq in question])] = wcpu
    
        if self._settings['print']:
            print(w)

        if self._settings['show'] or self._settings['save'] or self._settings['savepng']:
            qdf = pd.DataFrame(question_weights)
            sns.heatmap(qdf, cmap='YlGnBu')

            path = '{}/correlation_weights_{}_{}'.format(
                    self._settings['results_path'],
                    self._settings['pipeline']['data']['dataset'],
                    self._settings['pipeline']['data']['sequencer']
            )
            if self._settings['save']:
                plt.savefig('{}.svg'.format(path), format='svg')
            
            if self._settings['savepng']:
                plt.savefig('{}.png'.format(path), format='png')

            if self._settings['show']:
                plt.show()
            else:
                plt.close()

        return question_weights

    def cluster_question_weights(self, x):
        self._crawl()
        qw = self.get_correlation_weights(x)
        clustering = ClusteringPipeline(self._settings)
        cluster_results, best_silhouette_score = clustering.cluster([weight for weight in qw.values()])
        print(cluster_results, best_silhouette_score)

        if self._settings['print'] or self._settings['save']:
            scores = [k for k in cluster_results.keys()]
            scores.sort()

            if self._settings['print']:
                for s in scores:
                    print('Score: {},\nparameters: {}\n\n'.format(
                        s, cluster_results[s]
                    ))
            if self._settings['save']:
                path = '{}clustering_results.pkl'.format(self._settings['results_path'])
                with open(path, 'wb') as fp:
                    pickle.dump(cluster_results, fp)

        return cluster_results, cluster_results['clustering'][best_silhouette_score]

    def plot_clustered_question_weights(self, cluster_results, results):
        self._crawl()
        data = [d for d in cluster_results['parameters']['data']]
        labels = [l for l in results['labels']]

        path = '{}/clustered_correlation_weights_{}'.format(
            self._settings['results_path'],
            cluster_results['parameters']['algo']
        )

        min_val = np.min(data)
        max_val = np.max(data)

        for lab in np.unique(labels):
            indices = [i for i in range(len(labels)) if labels[i] == lab]
            lab_data = [data[idx] for idx in indices]

            labdf = pd.DataFrame(lab_data)
            sns.heatmap(labdf, cmap='YlGnBu', vmin=min_val, vmax=max_val)

            path_lab = '{}_{}'.format(path, lab)

            if self._settings['save']:
                plt.savefig('{}.svg'.format(path_lab), format='svg')
            if self._settings['savepng']:
                plt.savefig('{}.png'.format(path_lab), format='png')
            if self._settings['show']:
                plt.show()
            else:
                plt.close()




    