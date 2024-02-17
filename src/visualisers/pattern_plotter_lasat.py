from ast import pattern
import os
import re
import pickle
from typing import Pattern

import math
import numpy as np
import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt

from visualisers.stylers.full_sequences_styler import FullStyler
from visualisers.pattern_plotter import PatternPlotter
from pattern_mining.lasta.lasat_model import LasatModel

class LasatPatternPlotter(PatternPlotter):
    """This class plots nested crossvalidation results
    """
    
    def __init__(self, settings:dict):
        super().__init__(settings)
        self._settings = settings
        # self._styler = FullStyler(settings)

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
        # print('files', pattern_paths)
        pattern_paths = [ppath for ppath in pattern_paths if 'exclude' not in ppath]
        config_path = [ppath for ppath in pattern_paths if 'config.pkl' in ppath]
        data_path = [ppath for ppath in pattern_paths if 'data.pkl' in ppath]
        demographics_path = [ppath for ppath in pattern_paths if 'demographics.pkl' in ppath]
        kws = self._settings['experiment']['keyword']
        # print('kw', kws)
        for kw in kws:
            pattern_paths = [ppath for ppath in pattern_paths if kw in ppath]
        # print('second', pattern_paths)
        
        # Load pattern results
        pattern_paths = [ppath for ppath in pattern_paths if 'patterns_' in ppath]
        regex_pattern = re.compile(self._settings['experiment']['regex'])
        patterns_files = {}
        for ppath in pattern_paths:
            key = regex_pattern.findall(ppath)[0]
            patterns_files[key] = {
                'data': pd.read_csv(ppath),
                'result_path': ppath.split('/')[:-1],
                'attribute': self._get_data_label(key)
            }
            os.makedirs('/'.join(ppath.split('/')[:-1]), exist_ok=True)

        with open(config_path[0], 'rb') as fp:
            self._config = pickle.load(fp)
            self._lasat = LasatModel(self._config)
        with open(data_path[0], 'rb') as fp:
            self._data = pickle.load(fp)
        with open(demographics_path[0], 'rb') as fp:
            self._demographics = pickle.load(fp)  
            self._clean_demographics()      
        return patterns_files
        
    def _get_secundarypatterns_basedonprimary(self, main_patterns, secundary_group):
        """Given the main patterns, shows howw representative they are in the secundary group

        Args:
            main_patterns (_type_): list of patterns to look into 
            secundary_group (_type_): pattern results on the secundary group

        Returns:
            _type_: s-support of the secundary group for the main patterns
        """
        sec_patterns = {
            row['Pattern']: row['S-Support (S-Frequency %)'] for _, row in secundary_group.iterrows() if 'Pattern' in row and not math.isnan(row['S-Support (S-Frequency %)'])
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
        excludes = [
            'Dataset Info: ',
            'Applied Algorithm(s) Info: '
        ]
        # if 'Pattern' in main_patterns:
        if 'Pattern' in main_patterns:
            main_pats = [mp for mp in main_patterns['Pattern'] if mp not in excludes]
            main_supps = [ms for ms in main_patterns['S-Support (S-Frequency %)']]
        else:
            main_pats = []
            main_supps = []
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

        # else:
        #     print('no patterns for these groups')
        #     return [], []
        # # print('mp')
        # print(columns)
        

    def _get_dsm_feeling(self, dsm_value, i_support_main, i_support_group):
        if dsm_value <= self._settings['dsm']['pvalue']:
            if np.mean(i_support_main) > np.mean(i_support_group):
                return -1
            else:
                return 1
        else:
            return 0

    def get_dsm_heatmap(self, main_patterns, group_heatmap):
        if len(main_patterns) > 0:
            print('main_patterns', main_patterns[0])
        i_supports = self._lasat.compute_i_supports(
            self._data, self._demographics, 
            main_patterns, self._settings['heatmap_groups'][group_heatmap]
        )
        # print()
        # print('i supports')
        # print(i_supports)
        dsm_values = self._lasat.compute_dsm(i_supports, pvalue=self._settings['dsm']['pvalue'])
        dsm_heatmap_df = {
            'main': [0 for _ in range(len(main_patterns))]
        }
        for group in dsm_values:
            if group != 'main':
                dsm_heatmap_df[group] = [
                    self._get_dsm_feeling(
                        dsm_values[group][pattern], 
                        i_supports['main']['i-support'][pattern],
                        i_supports[group]['i-support'][pattern]
                    )
                   for pattern in main_patterns
                ]
        if self._settings['show']:
            self.get_heatmap(main_patterns, dsm_heatmap_df, -1, 1)
        # ATTENTION ONLY FOR ONE GROUP
        to_main = [ttest_pval for ttest_pval in dsm_heatmap_df[group] if ttest_pval == -1]
        to_both = [ttest_pval for ttest_pval in dsm_heatmap_df[group] if ttest_pval == 1]
        # print(main_patterns, dsm_heatmap_df)
        print('Significant for main: {}'.format(len(to_main)))
        print('In both but significant for {}: {}'.format(group, len(to_both)))
        return dsm_heatmap_df

    def get_overlap(self, main_patterns, others_dict):
        if 'Pattern' in main_patterns:
            main_pats = [pat for pat in main_patterns['Pattern']][:-2]
        else:
            main_pats = []
        for group in others_dict:
            
            if 'Pattern' in others_dict[group]:
                other_pats = [pat for pat in others_dict[group]['Pattern']][:-2]
            else:
                other_pats = []
            print('    {} are overlapping between the main group and {}.'.format(
                len(set(main_pats).intersection(set(other_pats))), group
            ))
            print('    {} are specific to the main group.'.format(
                len(set(main_pats).difference(set(other_pats)))
            ))
            print('    {} are specific to the {} group.'.format(
                len(set(other_pats).difference(set(main_pats))), group
            ))

    def get_patterns_isupport_plot(self, group_heatmap):
        patterns = self._settings['plot']['isupport_patterns']
        i_supports = self._lasat.compute_i_supports(
            self._data, self._demographics, 
            patterns, self._settings['heatmap_groups'][group_heatmap]
        )
        
        y = 0.95
        plt.figure(figsize=(
                        self._settings['plot']['isupport_figsize'][0],
                        self._settings['plot']['isupport_figsize'][1]
        ))
        # print(i_supports)
        for pattern in patterns:
            for group in i_supports:
                # print(i_supports)
                # print(i_supports['main']['i-support'][pattern])
                # print([y for _ in range(len(i_supports['main']['i-support'][pattern]))])
                if group == 'main':
                    plt.scatter(
                        i_supports['main']['i-support'][pattern], 
                        [y for _ in range(len(i_supports['main']['i-support'][pattern]))],
                        color=self._settings['plot']['colour']['main'], alpha=0.2,
                        # label='main'
                    )
                else:
                    plt.scatter(
                        i_supports[group]['i-support'][pattern], 
                        [y for _ in range(len(i_supports[group]['i-support'][pattern]))],
                        color=self._settings['plot']['colour']['difference'], alpha=0.2,
                        # label=group
                    )
                y+=0.1
            y+=0.90

        plt.ylim([0, y])
        plt.scatter(
                        i_supports['main']['i-support'][pattern], 
                        [y+10 for _ in range(len(i_supports['main']['i-support'][pattern]))],
                        color=self._settings['plot']['colour']['main'], alpha=0.2,
                        label='main'
        )
        plt.scatter(
                        i_supports[group]['i-support'][pattern], 
                        [y+10 for _ in range(len(i_supports[group]['i-support'][pattern]))],
                        color=self._settings['plot']['colour']['difference'], alpha=0.2,
                        label=group
        )
        plt.legend()
        plt.yticks(range(1, len(patterns) + 2), [*patterns, ''])
        plt.show()

    def get_patterns_isupport_density_plot(self, group_heatmap):
        patterns = self._settings['plot']['isupport_patterns']
        i_supports = self._lasat.compute_i_supports(
            self._data, self._demographics, 
            patterns, self._settings['heatmap_groups'][group_heatmap]
        )
        with open('./isupport.pkl', 'wb') as fp:
            pickle.dump(i_supports, fp)
        
        
        # print(i_supports)
        for pattern in patterns:
            print(pattern)
            plt.figure(figsize=(
                        self._settings['plot']['isupport_figsize'][0],
                        self._settings['plot']['isupport_figsize'][1]
            ))
            for group in i_supports:
                # print(i_supports[group]['i-support'][pattern])
                if group == 'main':
                    # print(i_supports['main']['i-support'][pattern])
                    maxbin = max(i_supports['main']['i-support'][pattern])
                    sns.histplot(
                        data=i_supports['main']['i-support'], x=pattern, kde=True, bins=np.arange(-0.5, maxbin+1, 1),
                        color=self._settings['plot']['colour']['main'], alpha=0.2,
                        label='main'
                    )
                else:
                    maxbin = max(i_supports['main']['i-support'][pattern])
                    # print(i_supports[group]['i-support'][pattern])
                    sns.histplot(
                        data=i_supports[group]['i-support'], x=pattern, kde=True, bins=np.arange(-0.5, maxbin+1, 1),
                        color=self._settings['plot']['colour']['difference'], alpha=0.2,
                        label=group
                    )

            plt.xlabel('i-support')
            plt.ylabel('#students')
            plt.legend()
            plt.show()


    def get_heatmaps(self):
        patterns_paths = self._crawl()

        keys = [k for k in patterns_paths]
        [patterns_paths[k]['attribute'].sort() for k in keys]
        attributes = ['_'.join(patterns_paths[k]['attribute']) for k in keys]
        # print('attributes', attributes)
        for group in self._settings['heatmap_groups']:
            print('Analysing: {} against {}'.format(
                self._settings['heatmap_groups'][group]['main'],
                self._settings['heatmap_groups'][group]['others'] 
            ))
            main = self._settings['heatmap_groups'][group]['main'].split('.')
            main.sort()
            main = '_'.join(main)
            main = attributes.index(main)
            main_patterns = patterns_paths[keys[main]]['data']
            #print(main_patterns)
            secundary_groups = [other.split('.') for other in self._settings['heatmap_groups'][group]['others']]
            [sg.sort() for sg in secundary_groups]
            secundary_groups = ['_'.join(sg) for sg in secundary_groups]
            secundary_attributes = [attributes.index(sg) for sg in secundary_groups]
            secundary_keys = [keys[sa] for sa in secundary_attributes]
            others = [patterns_paths[sidx]['data'] for sidx in secundary_keys]
            assert len(secundary_keys) == len(others)
            others_dict = {secundary_keys[i]: others[i] for i in range(len(secundary_keys))}


            # print(others_dict)
            results = {}
            dsm = {}
            main_pats, heatmain_patsmap_df = self.get_dict_heatmap(main_patterns, others_dict)
            with open('./main_pats.pkl', 'wb') as fp:
                pickle.dump(main_pats, fp)
            with open('./heatmain_patsmap_df.pkl', 'wb') as fp:
                pickle.dump(heatmain_patsmap_df, fp)
            # print(main_pats)
            if self._settings['s_support']:
                self.get_heatmap(main_pats, heatmain_patsmap_df)
            if self._settings['differences']:
                self.get_differences(main_pats, heatmain_patsmap_df)
            if self._settings['dsm_plot']:
                dsm = self.get_dsm_heatmap(main_pats, group)
            if self._settings['overlap']:
                results['overlap'] = self.get_overlap(main_patterns, others_dict)
            if self._settings['i-supports']:
                results['i_supports'] = self.get_patterns_isupport_plot(group)
                self.get_patterns_isupport_density_plot(group)
            print('...')
            print()

            return main_pats, heatmain_patsmap_df, dsm
                







            
        
        
    
                    
                
            
                