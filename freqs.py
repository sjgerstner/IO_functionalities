from argparse import ArgumentParser
from itertools import chain
import os
import pickle

import numpy as np
import pandas as pd
from pandas.io.formats.style import Styler
import torch

import neuron_choice
import plotting

parser = ArgumentParser()
parser.add_argument('--work_dir', default=None)
parser.add_argument('--neuroscope_dir', default='OLMo-7B-0424-hf_dolma-small')
parser.add_argument('--wcos_dir', default='wcos_casestudies')
parser.add_argument('--model', default='allenai/OLMo-7B-0424-hf')
parser.add_argument('--log', action='store_true')
parser.add_argument('--metric', default='summary_freq')#TODO support for other metrics
parser.add_argument('--subexperiments', nargs='+', default=['all'])
parser.add_argument('--layer_list', nargs='+', default=[0, 15, 31])
args = parser.parse_args()

if 'all' in args.subexperiments:
    subexps = ["layer_plots", "category_plots", "scatter_plots", "table", "selected"]
else:
    subexps = args.subexperiments

#tensor of frequency by neuron
SUMMARY_PATH = f'{args.work_dir}/neuroscope/results/{args.neuroscope_dir}/summary.pickle'
#TODO change file path to pt when ready
with open(SUMMARY_PATH, 'rb') as f:
    summary_dict = pickle.load(f)
freq_tensor = summary_dict[args.metric].cpu()#layer neuron
n_layers = freq_tensor.shape[0]
d_mlp = freq_tensor.shape[1]

PLOT_DIR = f'{args.work_dir}/plots/freq/{args.model}'
os.makedirs(PLOT_DIR, exist_ok=True)

#plotting by layer
LAYER_PATH = f'{PLOT_DIR}/{args.metric}_layers_{args.log}.pdf'
if "layer_plots" in subexps:#not os.path.exists(LAYER_PATH):
    plotting.aligned_histograms(
        [list(freq_tensor[layer]) for layer in range(n_layers)],
        [str(i) for i in range(n_layers)],
        f"{args.metric} by layer in {args.model}",
        LAYER_PATH,
        ncols=4,
        log=args.log,
        weighted=True,
    )

#plotting by category
CATEGORY_PATH = f'{PLOT_DIR}/{args.metric}_categories_{args.log}.pdf'
if "category_plots" in subexps or "table" in subexps:#not os.path.exists(CATEGORY_PATH):
    my_data = []
    for i,name in enumerate(neuron_choice.CATEGORY_NAMES):
        #lists of (layer neuron) tuples
        neuron_list, baseline_list = neuron_choice.neuron_choice(args, name)
        my_data.append([freq_tensor[index].item() for index in neuron_list])
        my_data.append([freq_tensor[index].item() for index in baseline_list])

    if "category_plots" in subexps:
        plotting.aligned_histograms(
            my_data,
            list(chain.from_iterable(
                (name, name+'_baseline') for name in neuron_choice.CATEGORY_NAMES
            )),
            f"{args.metric} of neurons in {args.model}",
            CATEGORY_PATH,
            ncols=2,
            log=args.log,
            weighted=True
        )
    if "table" in subexps:
        PICKLE_ALL = f'{PLOT_DIR}/{args.metric}_all.pickle'
        if not os.path.exists(PICKLE_ALL):
            #tensor of category by neuron
            PATH = f"{args.work_dir}/{args.wcos_dir}/results/{args.model_name}"
            with open(f"{PATH}/data.pickle", 'rb') as f:
                data = pickle.load(f)
            category_tensor = data['categories']#layer neuron
            #we have a tensor of frequencies (already above) a tensor of categories
            #later we want to choose subsets based on layer only (done above), category only,
            # or layer and category
            #for hist plots (first two cases) and for means (all cases)
            # and counts (last case, should be done by the wcos code)
            #TODO save means as separate dataframes
        else:
            pass#TODO
        PICKLE_MEANS = f'{PLOT_DIR}/{args.metric}_means.pickle'
        if not os.path.exists(PICKLE_MEANS):
            means = [np.mean(np.array(subset)) for subset in my_data]
            means = np.array(means).reshape((len(neuron_choice.CATEGORY_NAMES),2))
            table = pd.DataFrame(
                means, index=neuron_choice.CATEGORY_NAMES, columns=["true", "baseline"]
            )
            table.to_pickle(PICKLE_MEANS)
        else:
            table = pd.read_pickle(PICKLE_MEANS)
        styler = Styler(table, precision=3)
        styler.to_latex(
            f'{PLOT_DIR}/{args.metric}.tex', label=f"tab:{args.metric}",
            caption="""
            Activation frequencies by IO class.
            The second column represents random neurons taken from the same layers.
            """,
        )

#scatter plots
if "scatter_plots" in subexps or "selected" in subexps:
    #tensor of category by neuron
    PATH = f"{args.work_dir}/{args.wcos_dir}/results/{args.model}"
    with open(f"{PATH}/data.pickle", 'rb') as f:
        data = pickle.load(f)
    linout = data["linout"]
    absgateout = torch.abs(data["gateout"])
    absgatelin = torch.abs(data["gatelin"])

    #scatter plots
    SIMS = ["linout", "gateout", "gatelin"]
    data_by_layer = [
        {
            args.metric: freq_tensor[layer],
            "linout": linout[layer],
            "gateout": absgateout[layer],
            "gatelin": absgatelin[layer],
        }
        for layer in range(n_layers)
    ]
    NCOLS = 4
    if "scatter_plots" in subexps:
        for sim in SIMS:
            plotting.freq_sim_scatter(
                data_by_layer,
                (args.metric, sim),
                (int(np.ceil(n_layers/NCOLS)), NCOLS),
                f"{args.metric} vs. {sim} in {args.model}",
                f'{PLOT_DIR}/{args.metric}_{sim}.pdf'
            )
    if "selected" in subexps:
        plotting.freq_sim_scatter(
            data_by_layer,
            (args.metric, "linout"),
            arrangement = (1, len(args.layer_list)),
            suptitle = f"{args.metric} vs. linout in {args.model}",
            savefile = f'{PLOT_DIR}/{args.metric}_linout_selected.pdf',
            layer_list=args.layer_list,
        )
