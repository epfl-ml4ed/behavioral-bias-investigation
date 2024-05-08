import os
import logging
from tabnanny import check
import numpy as np
import pandas as pd
from scipy import stats
from utils.statistical import fischer_mean

import subprocess
from pattern_mining.miner import Miner

class LasatModel(Miner):

    def __init__(self, settings):
        super().__init__(settings)
        self._model_settings = self._settings['pm']['models']['lasat']
        self._name = 'lasat'
        self._notation = 'lasat'

        self._support_threshold = self._model_settings['support_threshold']
        self._out_path = '{}/patterns-mined/'.format(
            self._settings['experiment']['name']
        )
        os.makedirs(self._out_path, exist_ok=True)
        

    def _to_lasat_format(self, sequences:list) -> pd.DataFrame:
        """Transforms the data into the correct format for the apriori algorithm
        As implemented on 
        https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/#apriori-frequent-itemsets-via-the-apriori-algorithm


        Args:
            student_df (list of list): each external list represents a student. Each student
            is a list of the item that corresponds them. For example: 
            [
                [high_concentration, high_width, low_wavelength, high_greenred], # student 0
                [low_concentration, low_width, high_wavelength, high_greenred] # student 1
            ]

        Returns:
            student_item_data (dataframe): columns are are the different items, rows represent a student. 
            data[student, item] is a boolean of whether the student belong to this category
        """
        student_df = pd.DataFrame()
        student_df['sequences'] = sequences
        student_df['lid'] = list(range(len(student_df)))
        student_df = student_df[['sequences', 'lid']]

        path = '{}/input_data'.format(self._settings['experiment']['name'])
        os.makedirs(path, exist_ok=True)
        path = '{}/data.csv'.format(path)
        path = path.replace('/','//')
        student_df.to_csv(path, index=False)
        return path


    def get_patterns(self, sequences:list, prtcle='') -> pd.DataFrame:
        particule = prtcle.replace(', ', '').replace('ç', 'c')
        path_sequences = self._to_lasat_format(sequences)

        with open('./pattern_mining/DSM/config/Beerslaw.properties', 'r+') as file:
            data = file.readlines()

        # Write parameters
        data[1] = 'spm.minsupport = {} \n'.format(self._support_threshold)
        data[8] = 'in.seqfile.filename = {} \n'.format(path_sequences)
        data[-1] = 'out.seqresults.file.filename = {}/patterns_{}.csv'.format(self._out_path, particule)
        with open('./pattern_mining/DSM/config/Beerslaw.properties', 'w') as file:
            file.writelines( data )
        
        print(data)

        # Run patterns
        subprocess.call(["java", "-jar", "./pattern_mining/DSM/lasat1.jar", "--configfile", "./pattern_mining/DSM/config/Beerslaw.config"])
        results = pd.read_csv('{}/patterns_{}.csv'.format(self._out_path, particule))

        return results

    def compute_s_support(self, fre_patt, all_sequences):
        pattern = '; '.join(fre_patt)
        results = [pattern in a_s for a_s in all_sequences]
        print(fre_patt, np.sum(results) / len(all_sequences))
        return np.sum(results) / len(all_sequences)

    def compute_i_support(self, fre_patt, all_sequences):
        fre_patt = fre_patt.replace(' ->  ', '; ')
        # print('pat', fre_patt)
        # print('seq', all_sequences[0])
        results = [ss.count(fre_patt) for ss in all_sequences]
        # print('{} Computing I Supports {}'.format('*' * 20, '*' * 20))
        if fre_patt != [] and np.sum(results) == 0:
            print(fre_patt)
            print('a',all_sequences[0].split(';')[0])
        #     print(results[0])
        # print('res', results)
        # print('len', len(all_sequences))
        return results

    def dsm(self, sequences_a, sequences_b, patterns):
        patterns_significance = {}
        for pattern in patterns:
            i_a = self.compute_i_support(pattern, sequences_a)
            i_b = self.compute_i_support(pattern, sequences_b)
            
            patterns_significance[pattern] = {
                'a': {
                    'mean': np.mean(i_a),
                    'i-supports': i_a
                },
                'b': {
                    'mean': np.mean(i_b),
                    'i-supports': i_b
                },
                'p-value': fischer_mean(i_a, i_b)
            }

        return patterns_significance

    def mine_all(self, data, demographics):
        logging.info('Mining process starting...')
        results = {}
        results.update(
            self.mine_general(data, demographics)
        )
        results.update(
            self.mine_through_intersection(data, demographics)
        )
        # results.update(self.mine_through_intersection_association(data, demographics))
        return results

    def _composite_demographics(self, check_demo, student_demo):
        assert not ('.' in check_demo and '_' in check_demo)
        if '.' in check_demo:
            strings = check_demo.split('.')
        elif '_' in check_demo:
            strings = check_demo.split('_')
        else:
            strings = [check_demo]
        bools = []
        for s in strings:
            if s in student_demo:
                bools.append(1)
        return len(bools) == len(strings)
        
    def compute_i_supports(self, data, demographics, patterns, populations:dict):
        """Computes the i-support between the main population and the other populations.


        Args:
            data (list): data[n] is the sequence for student n
            demographics (list<dict>]): demographics[n] is a dictionary grouping the demographics for student n
            populations (dictionary):
                main: name of the attributes to compare with
                others: name of the attributes to compare with main

        [ x ] Take the demographics in other and do pipeline
        [ x ] Take the demographics in main and do pipeline
        [ x ] Create pipeline: take data, count the i-support for specific patterns
        [ x ] do dsm between each of the main and others
        [  ]  send back the heatmaps
        [  ] plot it with the heatmaps
        """
        demographics_string = [[str(v) for v in student.values()] for student in demographics]
        [student.sort() for student in demographics_string]
        demographics_string = ['_'.join(student) for student in demographics_string]

        i_supports = {}
        main_indices = [i for i in range(len(data)) if self._composite_demographics(populations['main'], demographics_string[i])]
        main_data = [data[idx] for idx in main_indices]
        # print('main indices', populations['main'])
        # print(main_indices)
        print('{} main_data length: {}'.format(' '*40, len(main_data)))
        i_supports['main'] = {
            'i-support': {pattern: self.compute_i_support(pattern, main_data) for pattern in patterns},
            'n': len(main_data)
        }
        for other in populations['others']:
            other_label = other.split('.')
            other_label.sort()
            other_label = '_'.join(other_label)
            other_indices = [i for i in range(len(data)) if self._composite_demographics(other_label, demographics_string[i])]
            other_data = [data[idx] for idx in other_indices]
            i_supports[other_label] = {
                'i-support': {pattern: self.compute_i_support(pattern, other_data) for pattern in patterns},
                'n': len(other_data)
            }
            print('{} length of the other groups: {}'.format(' '*40, i_supports[other_label]['n']))
        
        return i_supports
    
    def _t_test(self, a, b, pvalue=0.05):
        # print(a)
        # print(b)
        # print(fischer_mean(a, b, equal_var = False).pvalue)
        # print()
        return fischer_mean(a, b) #<= pvalue

    def compute_dsm(self, i_supports, pvalue=0.05):
        """Compute whether the patterns are different across the i-supports of different groups

        Args:
            i_supports (dict): 
                value_dict are like:
                    i-support : {pattern: [i-supports]},
                    n: len of the data
                main: value_dict
                group0: value-dict
                group1: value_dict
                etc.
        """
        dsm_values = {}
        for group in i_supports:
            if group != 'main':
                # print(group)
                dsm_values[group] = {
                    pattern: self._t_test(
                        i_supports['main']['i-support'][pattern],
                        i_supports[group]['i-support'][pattern],
                        pvalue
                    ) for pattern in i_supports['main']['i-support']
                }


        return dsm_values












    