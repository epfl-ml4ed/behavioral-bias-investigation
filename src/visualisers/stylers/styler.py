import os
import re
import yaml
import numpy as np

import bokeh
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.models import ColumnDataSource, Whisker
from bokeh.sampledata.autompg import autompg as df

class Styler:
    """Processes the plot config files to style the figures
    """
    
    def __init__(self, settings:dict):
        self._settings = settings
        
    def _get_maps(self):
        path = './visualisers/maps/colour.yaml'
        with open(path, 'r') as f:
            self._cm = yaml.load(f, Loader=yaml.FullLoader)
            
    def get_cm(self):
        return dict(self._cm)
    
    def get_lm(self):
        return dict(self._lm)
        
        
    def init_figure(self, xaxis:dict):
        p = figure(
            title=self._styler_settings['title'],
            sizing_mode=self._styler_settings['sizing_mode'],
            y_range=self._styler_settings['ystyle']['range'],
            x_range=self._styler_settings['xstyle']['range']
        )
        p.title.text_font_size = '25pt'
        p.xaxis.axis_label_text_font_size  = '20pt'
        p.yaxis.axis_label_text_font_size  = '20pt'
        
        
        xticks_labels = dict(zip(xaxis['ticks'], xaxis['labels']))
        xticks_labels = {float(xx)+0.000001:label for xx, label in xticks_labels.items()}
        p.xaxis.ticker = list(xticks_labels.keys())
        p.xaxis.major_label_overrides = xticks_labels
        p.xaxis.axis_label = self._styler_settings['xstyle']['label']
        p.yaxis.axis_label = self._styler_settings['ystyle']['label']
        p.xaxis.major_label_text_font_size = '20px'

        p.yaxis.ticker = list(np.arange(0, 1.1, 0.1))
        return p
    
    def _get_algo(self, path:str) -> str:
        if 'LSTM' in path:
            algo = 'LSTM'
        if 'lstmtorch' in path:
            algo = 'LSTM'
        if 'lstm' in path:
            algo = 'LSTM'
        if 'dkvmn' in path:
            algo = 'dkvmn'
        if 'dmkt' in path:
            algo = 'dmkt'
        if 'attention' in path:
            algo = 'LSTM'
        return algo
    
    def _get_feature(self, path:str) -> str:
        if 'colorado_capacitor' in path:
            feature = 'capacitor_colorado'
        if 'chemlab_beerslaw' in path:
            feature = 'chemlab_beerslaw'
        if 'x_generalised' in path:
            feature = 'generalised'
        if 'chemlab_baseline' in path:
            feature = 'chemlab_beerslaw'

        if 'lightbeer_capacitor' in path:
            feature = 'capacitor_colorado'
        if 'lightbeer_beerslaw' in path:
            feature = 'chemlab_beerslaw'
        print(path)
        return feature

    def _get_regex(self, path:str) -> str:
        path_regex = ''
        p = path.replace('//', '/')
        for reg in self._settings['plot_style']['xstyle']['regexes']:
            re_exp = re.compile(reg)
            exp = re_exp.findall(p)[0]
            path_regex += '{}_'.format(exp)
        path_regex = path_regex[:-1]
        return path_regex
    
    def _get_alpha(self, path:str) -> str:
        if 'reproduction' in path:
            return 0.3
        else:
            return 0.9
        
        
    def _algofeatures_plot_styling(self, paths:list) -> dict:
        # TODO: use colour map
        alphas = []
        colours = []
        labels = []
        linedashes = []
        
        labels_colours_alpha = {}
        for path in paths:
            alpha = self._get_alpha(path)
            alphas.append(alpha)
            algo = self._get_algo(path)
            feature = self._get_feature(path)
            colour = self._cm[algo][feature]
            colours.append(colour)
            label = algo 
            if 'reproduction' in path:
                label += ' (reproduction)'
                linedash = 'dotted'
            else:
                linedash = 'solid'
            linedashes.append(linedash)
            labels.append(label)
            labels_colours_alpha[label] = {
                'colour': colour,
                'alpha': alpha
            }
            
        plot_styling = {
            'alphas': alphas,
            'colours' : colours,
            'labels': labels,
            'labels_colours_alpha' : labels_colours_alpha,
            'linedashes' : linedashes
        }
        return plot_styling
    
    def get_plot_styling(self, paths:list) -> dict:
        if self._styler_settings['style'] == 'algo_features':
            return self._algofeatures_plot_styling(paths)
        