from ast import pattern
import os
import re
import pickle
from sre_constants import GROUPREF_LOC_IGNORE
import numpy as np
import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt

from visualisers.stylers.full_sequences_styler import FullStyler
from visualisers.pattern_plotter import PatternPlotter
from pattern_mining.apriori.AprioriModel import APrioriModel

class AprioriPatternPlotter(PatternPlotter):
    """This class plots nested crossvalidation results
    """
    
    def __init__(self, settings:dict):
        super().__init__(settings)
        self._settings = settings

    def _crawl(self):
        """Crawl the files from the experiment folder, and look into the results files containing the patterns


        Returns:
           pattern_files (dict): 
                key: determined by the regex
                values: 
                    data: patterns
                    result_paths: where the results can be saved
                    attributes: demographics/label attributes on which we mined the patterns
        """
        # crawl paths
        pattern_paths = []
        experiment_path = '../experiments/{}/'.format(self._settings['experiment']['name'])
        for (dirpath, _, filenames) in os.walk(experiment_path):
            files = [os.path.join(dirpath, file) for file in filenames]
            pattern_paths.extend(files)
        pattern_paths = [ppath for ppath in pattern_paths if 'exclude' not in ppath]
        kws = self._settings['experiment']['keyword']
        for kw in kws:
            pattern_paths = [ppath for ppath in pattern_paths if kw in ppath]
        pattern_paths = [ppath for ppath in pattern_paths if 'apriori' in ppath]
        config_path = [ppath for ppath in pattern_paths if 'config.pkl' in ppath]
        data_path = [ppath for ppath in pattern_paths if 'data.pkl' in ppath]
        demographics_path = [ppath for ppath in pattern_paths if 'demographics.pkl' in ppath]
        pattern_paths = [ppath for ppath in pattern_paths if 'results.pkl' in ppath]
        
        # Load pattern results
        assert len(pattern_paths) == 1
        with open(pattern_paths[0], 'rb') as fp:
            all_patterns = pickle.load(fp)
        patterns_files = {}
        for demo in all_patterns:
            if demo == 'general':
                patterns_files['general'] = {
                    'data': all_patterns['general'],
                    'attribute': self._get_data_label(demo),
                    'association': False
                }
            else:
                for attribute in all_patterns[demo]:
                    att_name = attribute.replace('ç', 'c')
                    df = all_patterns[demo][attribute]['patterns']
                    if type(df) is list:
                        df = pd.DataFrame()
                        df['support'] = []
                        df['itemsets'] = []
                    df = df.sort_values('support')
                    patterns_files[att_name] = {
                        'data': df,
                        'attribute': self._get_data_label(att_name),
                        'association': 'association' in demo
                    }
                    
        with open(config_path[0], 'rb') as fp:
            self._config = pickle.load(fp)
            self._apriori = APrioriModel(self._config)
        with open(data_path[0], 'rb') as fp:
            self._data = pickle.load(fp)
        with open(demographics_path[0], 'rb') as fp:
            self._demographics = pickle.load(fp)  
            self._clean_demographics()      
        return patterns_files

    def _frozen_itemset_to_string(self, frozenset):
        frozen = [its for its in frozenset]
        frozen.sort()
        frozen = ', '.join(frozen)
        return frozen
        
    def _get_secundarypatterns_basedonprimary(self, main_patterns, secundary_group):
        """Given the main patterns, shows howw representative they are in the secundary group

        Args:
            main_patterns (_type_): list of patterns to look into 
            secundary_group (_type_): pattern results on the secundary group

        Returns:
            _type_: s-support of the secundary group for the main patterns
        """
        secundary_group['string_itemsets'] = secundary_group['itemsets'].apply(self._frozen_itemset_to_string)
        sec_patterns = {
            row['string_itemsets']: row['support'] for _, row in secundary_group.iterrows()
        }

        column = []
        for patt in main_patterns:
            if patt not in sec_patterns:
                column.append(0)
            else:
                column.append(sec_patterns[patt])
        return column


    def get_dict_heatmap(self, main_patterns, other_patterns):
        """Get the dictionary ready for the heatmaps

        Args:
            main_patterns (_type_): list of patterns to compare
            other_patterns (_type_): pattern results for all secundary groups

        Returns:
            main_pats (list): list of patterns to compare
            columns (dictionary): 
                key: subpopulation
                value: s-support for the main pats
        """
        main_pats = [self._frozen_itemset_to_string(mp) for mp in main_patterns['itemsets']]
        main_supps = [ms for ms in main_patterns['support']]
        main_supps = [main_supps[i] for i in range(len(main_pats))]
        columns = {
            'main': main_supps
        }
        for other_pats in other_patterns:
            other_dict = self._get_secundarypatterns_basedonprimary(
                main_pats, other_patterns[other_pats]
            )
            columns[other_pats] = other_dict

        return main_pats, columns

    def get_overlap(self, main_patterns, others_dict):
        main_pats = [self._frozen_itemset_to_string(pat) for pat in main_patterns['itemsets']]
        print('There are {} patterns in the main group'.format(len(main_pats)))
        for group in others_dict:
            other_pats = [self._frozen_itemset_to_string(pat) for pat in others_dict[group]['itemsets']]
            print(
                '  I found {} patterns in the {} group. This resulted in: '.format(
                    len(other_pats), group
                )
            )
            print('    {} patterns overlapping between the main group and {}.'.format(
                len(set(main_pats).intersection(set(other_pats))), group
            ))
            print('    {} patterns specific to the main group.'.format(
                len(set(main_pats).difference(set(other_pats)))
            ))
            print('    {} patterns specific to the {} group.'.format(
                len(set(other_pats).difference(set(main_pats))), group
            ))

    def get_patterns_distribution(self, group_heatmap):
        """_summary_

        Args:
            *As given by self.get_dict_heatmap(main_patterns, other_pattern)*
            main_patterns (_type_): list of patterns to compare
            patterns_supports (dictionary): 
                key: subpopulation
                value: s-support for the main pats
        """
        patterns = self._settings['plot']['apriori_patterns']
        probs = self._apriori.compute_props(
            self._data, self._demographics,
            patterns, self._settings['heatmap_groups'][group_heatmap]
        )

        support_df = {
            'main': [probs['main']['props'][pattern] for pattern in patterns]
        }
        for group in group_heatmap['others']:
            support_df[group] = [
                probs[group]['props'][pattern] for pattern in patterns
            ]

        self.get_heatmap(patterns, support_df, 0, 1)



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
            if self._settings['overlap']:
                self.get_overlap(main_patterns, others_dict)
            if self._settings['distribution']:
                self.get_patterns_distribution(group)
                







            
        
        
    
                    
                
            
                