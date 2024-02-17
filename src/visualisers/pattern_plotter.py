from ast import pattern
import os
import re
import pickle
import numpy as np
import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt

from visualisers.stylers.full_sequences_styler import FullStyler
from pattern_mining.lasta.lasat_model import LasatModel

class PatternPlotter:
    """This class plots nested crossvalidation results
    """
    
    def __init__(self, settings:dict):
        self._settings = settings

    def _get_data_label(self, name:str):
        name = name.replace('patterns_', '')
        name = name.split('_')
        return name

    def _clean_demographics(self):
        new_demographics = [dict(demo) for demo in self._demographics]
        # self._config['pm']['demographics']['sa_gru'] = {0: 'sagru0', 1: 'sagru1'}
        for demo in self._config['pm']['demographics']:
            try:
                if self._config['pm']['demographics'][demo] != 'lam':
                    if ',' in demo:
                        demo_names = demo.split(',')
                        new_demos = [
                            self._config['pm']['demographics'][demo][student[demo_names[0]]].replace('ç', 'c') for student in self._demographics
                        ]
                        demo_label = demo.split(',')[0]
                        new_demos = [
                            self._config['pm']['demographics'][demo][student[demo_label]].replace('ç', 'c') for student in self._demographics
                        ]
                        
                    elif demo in self._demographics[0]:
                        new_demos = [
                            self._config['pm']['demographics'][demo][student[demo]].replace('ç', 'c') for student in self._demographics
                        ]
                else:
                    new_demos = [
                        student[demo].replace('ç', 'c') for student in self._demographics
                    ]

                [student.update({demo: new_demos[i_s]}) for i_s, student in enumerate(new_demographics)]
                [student.update({'general': 'general'}) for _, student in enumerate(new_demographics)]
            except KeyError: 
                # print('Could not process {}'.format(demo))
                continue
        
            except NameError:
                # print('Could not process {}'.format(demo))
                continue
        self._demographics = [dict(demo) for demo in new_demographics]
        
    def get_differences(self, patterns:list, dic_patterns:dict):
        """Compute the difference for each patterns between the main group and the others

        Args:
            patterns (list): list of patterns
            dic_patterns (dictionary):
                keys (str): subpopulation group
                values (1 x len(patterns) -> list): represenation value of the pattern for that subpopulation group 
        """
        differences = {
            'main': dic_patterns['main']
        }
        for key in dic_patterns:
            if key != 'main':
                differences[key] = [
                    (dic_patterns[key][idx] - dic_patterns['main'][idx]) for idx in range(len(dic_patterns['main']))
                ]
        self.get_heatmap(patterns, differences, -1, 1, 'difference')

    def get_heatmap(self, patterns: list, dic_patterns: dict, vmin=0, vmax=1, cmap='main'):
        """Create a heatmap comparing the values taken for each of the patterns

        Args:
            patterns (list): list of patterns
            dic_patterns (dictionary):
                keys (str): subpopulation group
                values (1 x len(patterns) -> list): represenation value of the pattern for that subpopulation group 
        """
        heatmap_df = pd.DataFrame(dic_patterns)
        heatmap_df.index = patterns

        plt.figure(figsize=(
            self._settings['plot']['figsize'][0],
            self._settings['plot']['figsize'][1]
        ))
        sns.heatmap(
            heatmap_df, annot=self._settings['plot']['annot'],
            cmap=self._settings['plot']['cmap'][cmap], vmin=vmin, vmax=vmax
        )
        plt.show()
        print('There are {} patterns'.format(len(heatmap_df)))

    def get_heatmaps(self):
        patterns_paths = self._crawl()

        keys = [k for k in patterns_paths]
        [patterns_paths[k]['attribute'].sort() for k in keys]
        attributes = ['_'.join(patterns_paths[k]['attribute']) for k in keys]

        for group in self._settings['heatmap_groups']:
            main = self._settings['heatmap_groups'][group]['main'].split('.')
            main.sort()
            main = '_'.join(main)
            main = attributes.index(main)
            main_patterns = patterns_paths[keys[main]]['data']

            secundary_groups = [other.split('.') for other in self._settings['heatmap_groups'][group]['others']]
            [sg.sort() for sg in secundary_groups]
            secundary_groups = ['_'.join(sg) for sg in secundary_groups]
            secundary_attributes = [attributes.index(sg) for sg in secundary_groups]
            secundary_keys = [keys[sa] for sa in secundary_attributes]
            others = [patterns_paths[sidx]['data'] for sidx in secundary_keys]
            assert len(secundary_keys) == len(others)
            others_dict = {secundary_keys[i]: others[i] for i in range(len(secundary_keys))}


            # print(others_dict)
            main_pats, heatmain_patsmap_df = self.get_dict_heatmap(main_patterns, others_dict)
            # print(main_pats)
            if self._settings['s_support']:
                self.get_heatmap(main_pats, heatmain_patsmap_df)
            if self._settings['differences']:
                self.get_differences(main_pats, heatmain_patsmap_df)
            if self._settings['dsm_plot']:
                self.get_dsm_heatmap(main_pats, group)
            if self._settings['overlap']:
                self.get_overlap(main_patterns, others_dict)
            if self._settings['i-supports']:
                self.get_patterns_isupport_plot(group)
                self.get_patterns_isupport_density_plot(group)
                







            
        
        
    
                    
                
            
                