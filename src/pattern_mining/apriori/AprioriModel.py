import logging
import numpy as np
import pandas as pd

from collections import Counter

from pattern_mining.miner import Miner
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

class APrioriModel(Miner):

    def __init__(self, settings):
        super().__init__(settings)
        self._model_settings = self._settings['pm']['models']['apriori']
        self._name = 'a_priori'
        self._notation = 'ap'

        self._support_threshold = self._model_settings['support_threshold']
        self._association_threshold = self._model_settings['association_threshold']
        

    def _to_apriori_format(self, student_df:list) -> pd.DataFrame:
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
        te = TransactionEncoder()
        student_item_data = te.fit(student_df).transform(student_df)
        student_item_data = pd.DataFrame(student_item_data, columns=te.columns_)
        return student_item_data

    def compute_props(self, fre_patt, all_sequences):
        fre_patts = fre_patt.split(', ')
        indices = []
        for patt in fre_patts:
            new_indices = [i for i in range(len(all_sequences)) if patt in all_sequences[i]]
            indices = [*indices, new_indices]

        counter = Counter(indices)
        present = [key for key in counter if counter[key] == len(fre_patts)]

        return len(present) / len(all_sequences)

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
        
    def compute_props(self, data, demographics, patterns, populations:dict):
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

        props = {}
        main_indices = [i for i in range(len(data)) if self.compute_props(populations['main'], demographics_string[i])]
        main_data = [data[idx] for idx in main_indices]
        # print('main indices', populations['main'])
        # print(main_indices)
        props['main'] = {
            'props': {pattern: self.compute_props(pattern, main_data) for pattern in patterns},
            'n': len(main_data)
        }
        for other in populations['others']:
            other_label = other.split('.')
            other_label.sort()
            other_label = '_'.join(other_label)
            other_indices = [i for i in range(len(data)) if self.compute_props(other_label, demographics_string[i])]
            other_data = [data[idx] for idx in other_indices]
            props[other_label] = {
                'props': {pattern: self.compute_props(pattern, other_data) for pattern in patterns},
                'n': len(other_data)
            }
        
        return props

    def _min_support(self, student_item_data:pd.DataFrame) -> float:
        """Computes the ideal minimum support for de data
        Implemented by @yasmineben510


        Args:
            student_item_data (dataframe): columns are are the different items, rows represent a student. 
            data[student, item] is a boolean of whether the student belong to this category

        Returns:t
            min support: the number which should be used for minimum support
        """
    
        total_items = len(student_item_data)

        #Computing the support of each item
        support_per_item = student_item_data.sum(axis=0)/total_items
        
        mean = support_per_item.mean() 
        std = support_per_item.std()
        
        min_support = mean - std
        assert min_support >= 0 and min_support <= 1
        return min_support

    def get_patterns(self, sequences:list, particule='') -> pd.DataFrame:
        # Patterns that are common in this dataset
        a_priori_format = self._to_apriori_format(sequences)
        apriori_sequences = apriori(a_priori_format, min_support=self._support_threshold, use_colnames=True)
        apriori_sequences = apriori_sequences[apriori_sequences['support'] >= self._support_threshold]
        return apriori_sequences

    def _contains_label(self, consequents):
        if len(consequents) > 1:
            return False
        for label in self._model_settings['association_consequent']:
            if label in consequents:
                return True
        return False

    def _mine_demographics_associations(self, data, demographics):
        results = {}
        demographic_groups = np.unique(demographics)
        for attribute in demographic_groups:
            logging.info('      - {}'.format(attribute))
            indices = [i for i in range(len(data)) if demographics[i] == attribute]
            att_data = [data[idx] for idx in indices]
            att_patterns = self.get_association(att_data)
            results[attribute] = {
                'patterns': att_patterns,
                'n': len(att_data)
            }
        
        return results

    def mine_through_intersection_association(self, data, demographics):
        logging.info('  association mining per group of attributes..')
        results = {}

        for intersection_group in self._settings['pm']['pipeline']['demographics']:
            logging.info('    - {}'.format(intersection_group))
            intersections = intersection_group.split('.')
            intersection_demographics = ['association' for _ in range(len(data))]
            for demographic_type in intersections:
                label_bool = False
                if demographic_type == self._model_settings['association_labels']:
                    label_bool = True
                if self._settings['pm']['demographics'][demographic_type] == 'lam':
                    demos = [student[demographic_type] for student in demographics]
                else:
                    type_short = demographic_type.split(',')[0]
                    demo_map = self._settings['pm']['demographics'][demographic_type]
                    demos = [demo_map[student[type_short]] for student in demographics]

                if label_bool:
                    data = [student + [demos[i_s]] for i_s, student in enumerate(data)]
                else:
                    intersection_demographics = ['{}_{}'.format(intersection_demographics[i], demos[i]) for i in range(len(data))]
            associations_results = self._mine_demographics_associations(data, intersection_demographics)
            results['association_{}'.format(intersection_group.replace('.', '-'))] = associations_results
            
            assert len(intersection_demographics) == len(data) and len(data) == len(demographics)

        return results

    def get_association(self, sequences:list):
        apriori_sequences = self.get_patterns(sequences)
        rules = association_rules(apriori_sequences, metric='confidence', min_threshold=self._association_threshold)
        rules = rules[rules['confidence'] > self._association_threshold]
        label_rules = rules[rules['consequents'].apply(lambda cons: self._contains_label(cons))]
        if len(label_rules) > 0:
            return label_rules
        else:
            return ['none']

    def mine_all(self, data, demographics):
        logging.info('Mining process starting...')
        results = {}
        results.update(
            self.mine_general(data, demographics)
        )
        results.update(
            self.mine_through_intersection(data, demographics)
        )
        results.update(self.mine_through_intersection_association(data, demographics))
        return results












    