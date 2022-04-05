import numpy as np
import matplotlib.pyplot as plt
from .. import utils

def _parse_kwargs(**kwargs):
    kwargs_list = [
        {'key': 'alpha', 'default': 0.25, 'type': float},
        {'key': 'color', 'default': '#1f77b4', 'type': str},
        {'key': 'dpi', 'default': 100, 'type': float},
        {'key': 'figsize', 'default': (6, 3), 'type': tuple},
        {'key': 'fill', 'default': False, 'type': bool},
        {'key': 'is_barplot', 'default': True, 'type': bool},
        {'key': 'linewidth', 'default': 2, 'type': float},
        {'key': 'title', 'default': 'Cross-Correlogram', 'type': str}
    ]
    kwargs = utils.check_kwargs_list(kwargs_list, **kwargs)

    return kwargs

def plot_cross_correlogram(cross_correlation, window_length, **kwargs):
    kwargs = _parse_kwargs(**kwargs)

    plt.figure(figsize=kwargs.get('figsize'), dpi=kwargs.get('dpi'))
    n_max = int((np.size(cross_correlation, 0) - 1) / 2)
    tau = window_length / n_max
    times = np.arange(-n_max*tau, (n_max+1)*tau, tau)

    if kwargs.get('is_barplot') is True:
        plt.bar(times, cross_correlation, width=tau*0.75, color=kwargs.get('color'))
    else:
        if kwargs.get('fill') is True:
            plt.fill_between(times, cross_correlation, facecolor=kwargs.get('color'), alpha=kwargs.get('alpha'))
        
        plt.plot(times, cross_correlation, linewidth=kwargs.get('linewidth'), color=kwargs.get('color'))

    plt.title(kwargs.get('title'))
    plt.xlabel('Time (s)')
    plt.ylabel('C(' + r'$\tau$' + ')')

    ax = plt.gca()
    ax.set_ylim(0, None)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return