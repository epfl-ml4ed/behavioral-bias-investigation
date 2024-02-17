import logging
import numpy as np


class Miner:
    def __init__(self, settings):
        self._settings = dict(settings)

    def _mine_demographics_patterns(self, data, demographics):
        results = {}
        demographic_groups = np.unique(demographics)
        for attribute in demographic_groups:
            logging.info('      - {}'.format(attribute))
            indices = [i for i in range(len(data)) if demographics[i] == attribute]
            att_data = [data[idx] for idx in indices]
            att_patterns = self.get_patterns(att_data, attribute)
            results[attribute] = {
                'patterns': att_patterns,
                'n': len(att_data)
            }
        
        return results

    def mine_general(self, data, demographics):
        logging.info('  mining on the general population..')
        results = {
            'general': {
                'patterns': self.get_patterns(data, 'general'),
                'n': len(data)
            }
            
        } 
        return results

    def mine_through_intersection(self, data, demographics):
        logging.info('  mining per group of attributes..')
        results = {}

        for intersection_group in self._settings['pm']['pipeline']['demographics']:
            logging.info('    - {}'.format(intersection_group))
            intersections = intersection_group.split('.')
            intersection_demographics = ['' for _ in range(len(data))]
            for demographic_type in intersections:
                if self._settings['pm']['demographics'][demographic_type] == 'lam':
                    demos = [student[demographic_type] for student in demographics]
                else:
                    type_short = demographic_type.split(',')[0]
                    demo_map = self._settings['pm']['demographics'][demographic_type]
                    demos = [demo_map[student[type_short]] for student in demographics]

                intersection_demographics = ['{}_{}'.format(intersection_demographics[i], demos[i]) for i in range(len(data))]
            intersection_demographics = [ids[1:] for ids in intersection_demographics]
            results[intersection_group.replace('.', '-')] = self._mine_demographics_patterns(data, intersection_demographics)
            assert len(intersection_demographics) == len(data) and len(data) == len(demographics)

        return results

    



            # demos = [inters ofr inters in intersections 

    

            





        


