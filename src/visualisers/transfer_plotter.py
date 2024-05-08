import os
import pickle

import numpy as np
import pandas as pd

from ml.gridsearches.gridsearch import GridSearch
from visualisers.stylers.full_sequences_styler import FullStyler

from bokeh.io import export_svg, export_png
from bokeh.plotting import output_file, show, save

class TransferPlotter:
    """Plots Transfer Results"""

    def __init__(self, settings):
        self._settings = dict(settings)
        self._styler = FullStyler(settings)

    def _crawl(self):
        # crawl paths
        xval_path = []
        experiment_path = '../experiments/' + self._settings['experiment']['name'] + '/'
        for (dirpath, _, filenames) in os.walk(experiment_path):
            files = [os.path.join(dirpath, file) for file in filenames]
            xval_path.extend(files)
        for kw in self._settings['experiment']['keyword']:
            xval_path = [xval for xval in xval_path if kw in xval]
        xval_path = [xval for xval in xval_path if 'exclude' not in xval]
        
        # Load xvals
        paths = []
        xvs = {}
        for xv in xval_path:
            with open(xv, 'rb') as fp:
                # print(xv)
                xvs[xv] = {
                    'data': pickle.load(fp)
                }

            paths.append('/'.join(xv.split('/')[:-1]))
    
        paths = [p for p in np.unique(paths)]
        # assert len(paths) == 1
        self._settings['result_path'] = '{}/overall_results/'.format(paths[0])
        os.makedirs(self._settings['result_path'], exist_ok=True)
        return xvs

    def plot_experiment(self, xvs):
        dots, parameters, boxplots = [], [], []
        xvs, x_axis = self._styler.get_x_styling(xvs)
        plot_styling = self._styler.get_plot_styling(x_axis['paths'])
        means = []
        for path in x_axis['paths']:
            xv = xvs[path]['data']
            d, p, b = self._create_dataframes(xv)
            print('****')
            print(path)
            print(b)
            print()
            dots.append(d)
            parameters.append(p)
            boxplots.append(b)
            means.append(b.iloc[0]['mean'])

        self._multiple_plots(dots, parameters, boxplots, x_axis, plot_styling)
        print(means)

    def _multiple_plots(self, dots:list, param:list, boxplot:list, xaxis:dict, plot_styling:dict):
        glyphs = {
            'datapoints': {},
            'upper_moustache': {},
            'lower_moustache': {},
            'upper_rect': {},
            'lower_rect': {}
        }
        p = self._styler.init_figure(xaxis)
        for i in range(len(dots)):
            x = xaxis['position'][i]
            colour = plot_styling['colours'][i]
            label = plot_styling['labels'][i]
            alpha = plot_styling['alphas'][i]
            styler = {'colour': colour, 'label': label, 'alpha': alpha}
            glyphs, p = self._styler.get_individual_plot(dots[i], param[i], boxplot[i], glyphs, x, styler, p)
        self._styler.add_legend(plot_styling, p)
        self._save(p)
        self._show(p)

    def _create_dataframes(self, gs: GridSearch):
        """Generates the dataframes used to plot the nested xval from a gridsearch object

        Args:
            gs ([type]): [description]

        Returns:
            [type]: [description]
        """
        # dots dataframe
        dots = {}
        params = []
        for fold in range(100):
            if fold in gs:
                dots[fold] = {}
                dots[fold]['data'] = gs[fold][self._settings['plot_style']['metric']]
                for parameter in gs[fold]['best_params']:
                    param = parameter.replace('_', ' ')
                    if 'score' not in param and 'fold' not in param and 'index' not in param:
                        dots[fold][param] = str(gs[fold]['best_params'][parameter])
                        params.append(param.replace('_', ' '))
                dots[fold]['fold'] = fold
        dots_df = pd.DataFrame(dots).transpose()
        
        
        # statistics
        q1 = float(dots_df['data'].quantile(q=0.25))
        q2 = float(dots_df['data'].quantile(q=0.5))
        q3 = float(dots_df['data'].quantile(q=0.75))
        mean = float(dots_df['data'].mean())
        std = float(dots_df['data'].std())
        iqr = q3 - q1
        upper = q3 + 1.5*iqr
        lower = q1 - 1.5*iqr
        
        # boxplot dataframe
        boxplot = pd.DataFrame()
        boxplot['q1'] = [q1]
        boxplot['lower_error'] = [mean - std]
        boxplot['median'] = [q2]
        boxplot['mean'] = [mean]
        boxplot['std'] = std
        boxplot['upper_error'] = [mean + std]
        boxplot['q3'] = [q3]
        boxplot['upper'] = [upper]
        boxplot['lower'] = [lower]
        
        print(dots_df)
        return dots_df, set(list(params)), boxplot

    def _save(self, p, extension_path=''):
        
        root_path = '{}/mode{}_plot{}_y{}'.format(
            self._settings['result_path'], self._settings['plot_style']['xstyle']['type'],
            self._settings['plot_style']['ystyle']['label'], self._settings['plot_style']['ystyle']['label']
        )
        if self._settings['plot_style']['xstyle']['type'] == 'groups':
            root_path += '_g{}'.format('-'.join(self._settings['plot_style']['xstyle']['groups']))
        root_path += '{}'.format(extension_path)

        if self._settings['saveimg']:
            path = '{}.svg'.format(root_path)
            p.output_backend = 'svg'
            export_svg(p, filename=path)    
            save(p)

        if self._settings['savepng']:
            path = '{}.png'.format(root_path)
            export_png(p, filename=path)    
            save(p)

        if self._settings['save']:
            path = '{}.html'.format(root_path)    
            output_file(path, mode='inline')
            save(p)
            
    def _show(self, p):
        if self._settings['show']:
            show(p)

    

    def plot(self):
        xvs = self._crawl()
        # print(xvs.keys())
        # for xv in xvs:
        # self.plot_experiment({xv:xvs[xv]})
        self.plot_experiment(xvs)