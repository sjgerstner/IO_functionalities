"""Code to create the plots for the RW functionalities paper"""

import itertools

import numpy as np
import torch

import matplotlib.pyplot as plt
import seaborn as sns

from utils import COMBO_TO_NAME

torch.set_grad_enabled(False)

DEVICE='cuda:0'

plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = "serif"

# CATEGORY_COLORS = {
#     (1,1,1):(1,1,0,1),
#     (1,1,0):(1,1,0,.5),
#     (1,0,1):(.5,.5,0,1),
#     (1,0,0):(.5,.5,0,0.5),
#     (0,1,1):(0,0,0,1),
#     (0,1,0):(0.25,.25,0.25,0.5),#new: (0,0,0,.5) grey
#     (0,0,1):(0.9,0.9,0.9,1),#new: (0,.5,0,1) greenish
#     (-1,1,1):(0,0,1,1),
#     (-1,1,0):(0,0,1,.5),
#     (-1,0,1):(0,.5,.5,1),
#     (-1,0,0):(0,.5,.5,0.5),
# }

CATEGORY_COLORS = {
    key: (
        max(0,key[0]) * (key[1]+1) *.5,
        .5 if key[1]==0 else max(0,key[0]),
        max(0,-key[0]) * (key[1]+1)/2,
        (key[2]+1)/2,
    )
    for key in COMBO_TO_NAME
}
CATEGORY_COLORS[(0,0,1)]=(0.9,0.9,0.9,1)

SHORT_TO_LONG = {
    "gatelin":"$cos(w_{gate}, w_{in})$",
    "gateout":"$cos(w_{gate}, w_{out})$",
    "linout":"$cos(w_{in}, w_{out})$",
    "summary_freq": "Frequency of gate>0",
}

def my_survey(
    results, model_name, names_and_colors=CATEGORY_COLORS
):
    """
    Parameters
    ----------
    results : dict
        A mapping from cosine combos (as in COMBO_TO_NAME)
        to tensors of layerwise counts
        (i.e. tensors of shape (layer) and dtype int)
        (typically output of utils.layerwise_count())
    model_name: str
    category_names : list of str
        The category labels.
    category_colors: list of tuples of 4 floats
        The rgba colors corresponding to the categories
    """
    #my preprocessing:
    # labels = list(results.keys())
    # data = np.array(list(results.values()))
    labels = list(range(results[(1,1,1)].numel()))#[0,...,31] if 32 layers
    labels = [f'Layer {n}' for n in labels]
    data = np.array(torch.stack(list(results.values()), dim=-1).cpu())

    data_cum = data.cumsum(axis=1)#cumsum over categories

    fig, ax = plt.subplots()
    fig.set_figwidth(6.75)
    fig.set_figheight(6.75)
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (key, color) in enumerate(names_and_colors.items()):
    #for key in COMBO_TO_NAME:
        widths = data[:,i]
        starts = data_cum[:, i] - widths
        rects = ax.barh(labels, widths, left=starts,
                        #height=0.5,
                        label=COMBO_TO_NAME[key], color=color)
        if any(widths>700):
            r, g, b, _ = color
            text_color = 'black' if (r>.5 or g>.5) else 'white'#TODO
            ax.bar_label(rects, label_type='center', color=text_color,
                         #fontsize='xx-large'
                         )
    ax.legend(
        #ncols=len(category_names),
        bbox_to_anchor=(1,1),
        loc='upper left',
        #fontsize='xx-large'
        )

    ax.set_title(model_name)#TODO do we want that?

    return fig, ax

def wcos_plot(data, layer_list, arrangement, model_name):
    fig, axs = plt.subplots(arrangement[0], arrangement[1],
                        sharex=True,
                        sharey=True,
                        constrained_layout=True,
                        )
    fig.set_figwidth(1.35*(arrangement[1]+1))
    fig.set_figheight(1.35*arrangement[0]+1)
    axs_list = axs.ravel().tolist()
    for i,layer in enumerate(layer_list):
        ax=axs_list[i]
        ax.set_xlim(-1.,1.)
        ax.set_ylim(-1.,1.)
        ax.set_aspect(1, share=True)
        #actual data
        scatter = ax.scatter(
            data["gateout"][layer], data["linout"][layer], c=data["gatelin"][layer],
            vmin=0., vmax=1., linewidths=0, s=.25, rasterized=True
            )
        #standard normal randomness regions (e.g. red, dotted)
        ax.axhline(
            y=data["beta"][0], color="red", linestyle="dotted", linewidth=.25
        )
        ax.axhline(
            y=data["beta"][1], color="red", linestyle="dotted", linewidth=.25
        )
        ax.axvline(
            x=data["beta"][0], color="red", linestyle="dotted", linewidth=.25
        )
        ax.axvline(
            x=data["beta"][1], color="red", linestyle="dotted", linewidth=.25
        )
        #layer-specific randomness regions
        ax.axhline(
            y=data["randomness"]["linout"][layer,0].item(),
            color="red", linestyle="dashed", linewidth=.25
        )
        ax.axhline(
            y=data["randomness"]["linout"][layer,1].item(),
            color="red", linestyle="dashed", linewidth=.25
        )
        ax.axvline(
            x=data["randomness"]["gateout"][layer,0].item(),
            color="red", linestyle="dashed", linewidth=.25
        )
        ax.axvline(
            x=data["randomness"]["gateout"][layer,1].item(),
            color="red", linestyle="dashed", linewidth=.25
        )
        #title
        ax.set_title(f"Layer {layer}", fontsize=10)
        colorbar = fig.colorbar(scatter, ax=ax, label="$cos(w_{gate}, w_{in})$")
        #standard normal and layer-specific randomness regions for gate-in
        colorbar.ax.plot(
            .5, data["beta"][1], color="red", marker=".", markersize=.25,
        )
        colorbar.ax.axhline(
            y=data["randomness"]["gatelin"][layer].item(), color="red", linewidth=.25,
        )
    fig.supxlabel("$cos(w_{gate}, w_{out})$", fontsize=10)
    fig.supylabel("$cos(w_{in}, w_{out})$", fontsize=10)

    fig.suptitle(model_name, fontsize=10)

    return fig, axs

def plot_boxplots(data, model_name):
    n_layers = data['gatelin'].shape[0]
    fig, axs = plt.subplots(
        nrows=3,
        ncols=1,
        sharex=True,
    )
    fig.set_figwidth(6.75)
    fig.set_figheight(4.5)
    for i, (k,v) in enumerate(SHORT_TO_LONG.items()):
        mydata = data[k]
        if 'gate' in k:
            mydata = torch.abs(mydata)
            v = '$|' + v.strip('$') + '|$'
            axs[i].set_ylim(0.,1.)
        else:
            axs[i].set_ylim(-1.,1.)
        #zero-based indexing for consistency
        axs[i].boxplot(
            mydata.T,
            tick_labels=[str(i) for i in range(n_layers)],
            flierprops={'rasterized':True},
        )
        axs[i].set_ylabel(v, fontsize=10)
    fig.supxlabel('Layer', fontsize=10)
    fig.suptitle(model_name, fontsize=10)
    return fig, axs

def plot_all_medians(model_to_medians_dict):
    """make one plot with the median cos(w_in,w_out) similarities across layers of all models"""
    line_styles = ['solid', 'dotted']
    fig, ax = plt.subplots()
    fig.set_figwidth(6.75)
    fig.set_figheight(2)
    ax.set_xlim(0.,1.)
    ax.set_ylim(-1.,1.)
    ax.axhline(color='grey')
    ax.set_xlabel('Layer (relative to network depth)')
    ax.set_ylabel('median $cos(w_{in},w_{out})$')
    for i, (key,value) in enumerate(model_to_medians_dict.items()):
        x = np.linspace(0,1,value.size(dim=0))
        lines = ax.plot(x, value, label=key)
        lines[0].set_linestyle(line_styles[(i//10)])
    ax.legend(
        bbox_to_anchor=(1,1),
        loc='upper left',
    )
    return fig, ax

def histogram_subplot(ax, diff_nonzero, neuron_subset_name, **kwargs):
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
        axs_list[i] = histogram_subplot(axs_list[i], list_data[i], subtitles[i], **kwargs)
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

# def _plot_class_changes(ax, pair_to_nchanges_dict):
#     mat = np.full((11,11), np.nan)
#     for key in pair_to_nchanges_dict:
#         mat[key] = pair_to_nchanges_dict[key]
#     rowtotals = einops.reduce(mat, 'from to -> from 1', 'sum')
#     mat /= rowtotals
#     #don't show the diagonal
#     mat_nodiag = mat.copy()
#     for i in range(11):
#         mat_nodiag[i,i]=np.nan
#     ax.imshow(mat_nodiag)
#     #write all the probs including in the diagonal
#     for i in range(11):
#         for j in range(11):
#             _text = ax.text(j, i, f"{mat[i, j]:.2f}",
#                         ha="center", va="center",
#                         color='black' if (j==i and mat[i,j]!=np.nan) else "w",
#                         )
#     return ax

# def plot_class_changes_one(pair_to_nchanges_dict, model_name, stepsize):
#     fig, ax = plt.subplots()
#     fig.set_figwidth(6.75)
#     fig.set_figheight(6.75)
#     ax = _plot_class_changes(ax, pair_to_nchanges_dict)
#     ax.set_xticks(range(11), CATEGORY_COLORS.keys(),
#                   rotation=45, ha="right", rotation_mode="anchor")
#     ax.set_yticks(range(11), CATEGORY_COLORS.keys())
#     ax.set_ylabel('from class...', fontsize=10)
#     ax.set_xlabel('to class...', fontsize=10)
#     ax.set_title(f"{model_name}, transitions after {stepsize} steps")
#     return fig, ax

# def plot_class_changes_several(dict_of_dicts, arrangement, title):
#     fig, axs = plt.subplots(
#         arrangement[0],
#         arrangement[1],
#         sharex=True,
#         sharey=True,
#         constrained_layout=True
#     )
#     fig.set_figheight(6.75*arrangement[0])
#     fig.set_figwidth(6.75*arrangement[1])
#     axs_list = axs.ravel().tolist()
#     for i,key in enumerate(dict_of_dicts.keys()):
#         ax=axs_list[i]
#         ax = _plot_class_changes(ax, dict_of_dicts[key])
#         ax.set_title(str(key), fontsize=10)
#     axs_list[-1].set_xticks(range(11), CATEGORY_COLORS.keys(),
#                            rotation=45, ha="right", rotation_mode="anchor")
#     axs_list[-1].set_yticks(range(11), CATEGORY_COLORS.keys())
#     fig.supylabel("from class...", fontsize=10)
#     fig.supxlabel("to class...", fontsize=10)
#     fig.suptitle(title)
#     return fig, axs
