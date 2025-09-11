"""Code to create the plots for the RW functionalities paper"""

import numpy as np
import matplotlib.pyplot as plt

import torch
#import einops

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

NICKNAME_TO_FORMULA = {
    "gatelin":"$cos(w_{gate}, w_{in})$",
    "gateout":"$cos(w_{gate}, w_{out})$",
    "linout":"$cos(w_{in}, w_{out})$",
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
    labels = list(range(results[(1,1,1)].shape))#['0',...,'31'] if 32 layers
    data = np.array(torch.stack(results.values(), dim=-1))

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
    for i, (k,v) in enumerate(NICKNAME_TO_FORMULA.items()):
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
