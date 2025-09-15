import itertools
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

SHORT_TO_LONG = {
    "summary_freq": "Frequency of gate>0",
    "linout": '$\cos(W_{in}, W_{out})$',
    "gateout": '$|\cos(W_{gate}, W_{out})|$',
    "gatelin": '$|\cos(W_{gate}, W_{in})|$',
}

def my_subplot(ax, diff_nonzero, neuron_subset_name, **kwargs):
    ax.hist(diff_nonzero, **kwargs)
    ax.set_title(neuron_subset_name.replace(' ', '\n'))
    return ax

def aligned_histograms(list_data, subtitles, suptitle, savefile, ncols=1, n_bins=None, weighted=False, **kwargs):
    nplots = len(list_data)
    nrows = int(np.ceil(nplots/ncols))
    #ncols = len(list_list_data[0])
    #getting common bin sizes
    kwargs['bins'] = plt.hist(
        np.array(list(itertools.chain.from_iterable(list_data))),
        bins=n_bins,
    )[1]
    #plotting
    fig, axs = plt.subplots(
        nrows, ncols,
        sharex=True, sharey=True,
        layout='constrained',
    )
    fig.set_figwidth(max(4, 1.35*ncols))
    fig.set_figheight(1.35*nrows)
    axs_list = axs.ravel().tolist()
    for i in range(nplots):
        if weighted:
            kwargs['weights'] = np.ones(len(list_data[i])) / len(list_data[i])
        axs_list[i] = my_subplot(axs_list[i], list_data[i], subtitles[i], **kwargs)
    fig.suptitle(suptitle)
    fig.savefig(savefile, bbox_inches='tight')
    plt.close()

def freq_sim_scatter(ax, data, x, y, title, cbar=True,
                     #cbar_ax=None,
                     weighted=False, vmax=None):
    """
    "Scatter plot" (technically, a bivariate histogram)
    of sparsity (x) vs in_out_sim (y).
    Adapted from Gurnee et al. 2024, Universal Neurons:
    https://github.com/wesg52/universal-neurons/blob/main/paper_notebooks/mysteries.ipynb
    
    Args:
    ax: an already created axis
    data: a pandas dataframe, dict or similar
    x, y: two keys in data
    """
    if weighted:
        weights=np.ones(len(data[x]))/len(data[x])
        if vmax is None:
            vmax=1.
    else:
        weights=None
    ax = sns.histplot(
        data, x=x, y=y, ax=ax, cbar=cbar, bins=100,
        weights=weights, vmax=vmax,
        #cbar_ax=cbar_ax,
        cbar_kws={'orientation': 'horizontal', 'pad':0.02}
    )
    # label the colorbar
    #cbar = sns_plot.collections[0].colorbar
    #cbar.ax.set_xlabel('neuron count')

    # Move x-axis labels and ticks to the top
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()

    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.set_title(title)

    corr = np.corrcoef(data[x], data[y])[0, 1]
    m, b = np.polyfit(data[x], data[y], 1)

    ax.plot(data[x], m*data[x] + b, color='red', lw=0.8,alpha=0.8, label=f'corr: {corr:.2f}')
    ax.legend(loc='upper right', fontsize='small')


    ax.set_ylim(-1, 1)
    ax.set_xlim(-0.02, 1)
    ax.grid(alpha=0.3, linestyle='--')

    return ax

def freq_sim_scatter_by_layer(
    data_by_layer, keys, arrangement, suptitle, savefile, layer_list=None,
):
    """
    A figure containing, for each layer, a "scatter plot" (technically, a bivariate histogram)
    of sparsity (x) vs in_out_sim (y).
    Adapted from Gurnee et al. 2024, Universal Neurons:
    https://github.com/wesg52/universal-neurons/blob/main/paper_notebooks/mysteries.ipynb

    Args:
        data_by_layer:
            list of datasets (e.g. pandas dataframes or dictionaries),
            each corresponding to a layer and having keys x and y
        keys (tuple): keys for the dataset
        arrangement (tuple): number of rows and columns
    """
    if layer_list:
        n_layers = len(layer_list)
    else:
        n_layers = len(data_by_layer)
        layer_list = range(n_layers)

    #precompute vmax
    vmaxs = np.zeros(n_layers)
    for i,layer in enumerate(layer_list):
        data = data_by_layer[layer]
        vmaxs[i] = (np.max(np.histogram2d(data[keys[0]], data[keys[1]], bins=100)[0]))
    vmax = np.max(vmaxs)

    fig, axes = plt.subplots(
        arrangement[0], arrangement[1], #figsize=(12, 4),
        sharex=True, sharey=True,
        layout='constrained',
    )
    fig.set_figwidth(4*arrangement[1])
    fig.set_figheight(4*arrangement[0])

    axs_list = axes.ravel().tolist()
    #cbar_ax = fig.add_axes([.91, .3, .03, .4])
    for i, layer in enumerate(layer_list):
        data = data_by_layer[layer]
        axs_list[i] = freq_sim_scatter(
            axs_list[i], data, x=keys[0], y=keys[1],
            title=f'Layer {layer}', cbar=(i==0), weighted=False,
            vmax=vmax,
            #cbar_ax=cbar_ax if i==0 else None
        )

    # make y label larger
    fontsize=11.5
    fig.supxlabel(SHORT_TO_LONG[keys[0]], fontsize=fontsize)
    fig.supylabel(SHORT_TO_LONG[keys[1]], fontsize=fontsize)

    fig.suptitle(suptitle)

    #fig.tight_layout(rect=[0, 0, .9, 1])
    plt.savefig(savefile, dpi=150)
