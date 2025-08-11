"""Code to create the plots for the IO functionalities paper"""

import numpy as np
import matplotlib.pyplot as plt

import torch
import einops

torch.set_grad_enabled(False)

DEVICE='cuda:0'

plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = "serif"

CATEGORY_COLORS = {
    "enrichment":(1,1,0,1),
    "atypical enrichment":(1,1,0,.5),
    "conditional enrichment":(.5,.5,0,1),
    "atypical conditional enrichment":(.5,.5,0,0.5),
    "proportional change":(0,0,0,1),
    "atypical proportional change":(0.25,.25,0.25,0.5),
    "orthogonal output":(0.9,0.9,0.9,1),
    "depletion":(0,0,1,1),
    "atypical depletion":(0,0,1,.5),
    "conditional depletion":(0,.5,.5,1),
    "atypical conditional depletion":(0,.5,.5,0.5),
}

NICKNAME_TO_FORMULA = {
    "gatelin":"$cos(w_{gate}, w_{in})$",
    "gateout":"$cos(w_{gate}, w_{out})$",
    "linout":"$cos(w_{in}, w_{out})$",
}

def survey(results, model_name, names_and_colors=CATEGORY_COLORS):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    model_name: str
    category_names : list of str
        The category labels.
    category_colors: list of tuples of 4 floats
        The rgba colors corresponding to the categories
    """
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)

    fig, ax = plt.subplots()
    fig.set_figwidth(6.75)
    fig.set_figheight(6.75)
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(names_and_colors.items()):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh(labels, widths, left=starts,
                        #height=0.5,
                        label=colname, color=color)
        if any(widths>700):
            r, g, b, _ = color
            text_color = 'black' if (r>.5 or g>.5) else 'white'
            ax.bar_label(rects, label_type='center', color=text_color,
                         #fontsize='xx-large'
                         )
    ax.legend(
        #ncols=len(category_names),
        bbox_to_anchor=(1,1),
        loc='upper left',
        #fontsize='xx-large'
        )

    ax.set_title(model_name)

    return fig, ax

def wcos_plot(layer_list, arrangement, data, model_name):
    fig, axs = plt.subplots(arrangement[0], arrangement[1],
                        sharex=True,
                        sharey=True,
                        constrained_layout=True,
                        )
    fig.set_figwidth(1.35*(arrangement[1]+1))
    fig.set_figheight(1.35*arrangement[0])
    axs_list = axs.ravel().tolist()
    for i,layer in enumerate(layer_list):
        ax=axs_list[i]
        ax.set_xlim(-1.,1.)
        ax.set_ylim(-1.,1.)
        ax.set_aspect(1, share=True)
        #actual data
        scatter = ax.scatter(
            data["gateout"][layer], data["linout"][layer], c=data["gatelin"][layer],
            vmin=-1., vmax=1., linewidths=0, s=.25, rasterized=True
            )
        #standard normal randomness regions (e.g. red, dotted)
        ax.axhline(y=data["beta"][0], color="red", linestyle="dotted")
        ax.axhline(y=data["beta"][1], color="red", linestyle="dotted")
        ax.axvline(x=data["beta"][0], color="red", linestyle="dotted")
        ax.axvline(x=data["beta"][1], color="red", linestyle="dotted")
        #layer-specific randomness regions
        ax.axhline(y=data["randomness"]["linout"][layer,0], color="red", linestyle="dashed")
        ax.axhline(y=data["randomness"]["linout"][layer,1], color="red", linestyle="dashed")
        ax.axvline(x=data["randomness"]["gateout"][layer,0], color="red", linestyle="dashed")
        ax.axvline(x=data["randomness"]["gateout"][layer,1], color="red", linestyle="dashed")
        #title
        ax.set_title(f"Layer {layer}", fontsize=10)
        colorbar = fig.colorbar(scatter, ax=ax, label="$cos(w_{gate}, w_{in})$")
        #standard normal and layer-specific randomness regions for gate-in
        colorbar.axhline(y=data["beta"][2], color="red", linestyle="dotted")
        colorbar.axhline(y=data["randomness"]["gatelin"][layer], color="red", linestyle="dashed")
    fig.supxlabel("$cos(w_{gate}, w_{out})$", fontsize=10)
    fig.supylabel("$cos(w_{in}, w_{out})$", fontsize=10)

    fig.suptitle(model_name, fontsize=10)

    return fig, axs

def plot_quartiles(data, model_name):
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

def _plot_class_changes(ax, pair_to_nchanges_dict):
    mat = np.full((11,11), np.nan)
    for key in pair_to_nchanges_dict:
        mat[key] = pair_to_nchanges_dict[key]
    rowtotals = einops.reduce(mat, 'from to -> from 1', 'sum')
    mat /= rowtotals
    #don't show the diagonal
    mat_nodiag = mat.copy()
    for i in range(11):
        mat_nodiag[i,i]=np.nan
    ax.imshow(mat_nodiag)
    #write all the probs including in the diagonal
    for i in range(11):
        for j in range(11):
            _text = ax.text(j, i, f"{mat[i, j]:.2f}",
                        ha="center", va="center",
                        color='black' if (j==i and mat[i,j]!=np.nan) else "w",
                        )
    return ax

def plot_class_changes_one(pair_to_nchanges_dict, model_name, stepsize):
    fig, ax = plt.subplots()
    fig.set_figwidth(6.75)
    fig.set_figheight(6.75)
    ax = _plot_class_changes(ax, pair_to_nchanges_dict)
    ax.set_xticks(range(11), CATEGORY_COLORS.keys(),
                  rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(11), CATEGORY_COLORS.keys())
    ax.set_ylabel('from class...', fontsize=10)
    ax.set_xlabel('to class...', fontsize=10)
    ax.set_title(f"{model_name}, transitions after {stepsize} steps")
    return fig, ax

def plot_class_changes_several(dict_of_dicts, arrangement, title):
    fig, axs = plt.subplots(
        arrangement[0],
        arrangement[1],
        sharex=True,
        sharey=True,
        constrained_layout=True
    )
    fig.set_figheight(6.75*arrangement[0])
    fig.set_figwidth(6.75*arrangement[1])
    axs_list = axs.ravel().tolist()
    for i,key in enumerate(dict_of_dicts.keys()):
        ax=axs_list[i]
        ax = _plot_class_changes(ax, dict_of_dicts[key])
        ax.set_title(str(key), fontsize=10)
    axs_list[-1].set_xticks(range(11), CATEGORY_COLORS.keys(),
                           rotation=45, ha="right", rotation_mode="anchor")
    axs_list[-1].set_yticks(range(11), CATEGORY_COLORS.keys())
    fig.supylabel("from class...", fontsize=10)
    fig.supxlabel("to class...", fontsize=10)
    fig.suptitle(title)
    return fig, axs
