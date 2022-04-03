import numpy as np
import matplotlib.pyplot as plt
from .. import utils

def _parse_kwargs(**kwargs):
    kwargs_list = [
        {'key': 'color', 'default': '#1f77b4', 'type': str},
        {'key': 'dpi', 'default': 100, 'type': float},
        {'key': 'figsize', 'default': (6, 3), 'type': tuple},
        {'key': 'title', 'default': 'ISI Histogram', 'type': str},
        {'key': 'range', 'default': None, 'type': tuple}
    ]
    kwargs = utils.check_kwargs_list(kwargs_list, **kwargs)

    return kwargs

def plot_ISI_hist(ISI, bins, **kwargs):
    kwargs = _parse_kwargs(**kwargs)
    plt.figure(figsize=kwargs.get('figsize'), dpi=kwargs.get('dpi'))

    plt.hist(ISI, bins=bins, range=kwargs.get('range'), color=kwargs.get('color'))
    
    plt.title(kwargs.get('title'))
    plt.xlabel('ISI (s)')
    plt.ylabel('Spikes Count')

    ax = plt.gca()
    ax.set_xlim(0, None)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return