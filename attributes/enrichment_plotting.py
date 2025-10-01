"""Plotting code from Geva et al., Dissecting Recall:
https://github.com/google-research/google-research/tree/master/dissecting_factual_predictions

Copyright 2023 Google LLC.
Licensed under the Apache License, Version 2.0 (the "License");
"""

from argparse import ArgumentParser
import os

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(#context="notebook",
        rc={
            "font.size":16,
            # "axes.titlesize":16,
            # "axes.labelsize":16,
            # "xtick.labelsize": 16.0,
            # "ytick.labelsize": 16.0,
            # "legend.fontsize": 16.0
            },
        style='whitegrid')
palette_ = sns.color_palette("Set1")
palette = palette_[2:5] + palette_[7:]

parser = ArgumentParser()
# parser.add_argument('--model', default='allenai/OLMo-7B-0424-hf')
# parser.add_argument('--topk', default=50)
# parser.add_argument('--intervention_type', default=None)
# parser.add_argument('--activation_location', default='mlp.hook_pre')
parser.add_argument('--work_dir')
args = parser.parse_args()

OUT_DIR = f'{args.work_dir}/se'
DF_PATH = f'{OUT_DIR}/logitlens.pickle'
tmp = pd.read_pickle(DF_PATH)

if tmp["neuron_subset_name"].isna().any():
    tmp["neuron_subset_name"] = tmp["neuron_subset_name"].fillna('clean')
    tmp.to_pickle(DF_PATH)

all_subset_names = tmp.neuron_subset_name.unique()

for name in all_subset_names:
    if name=='clean' or '_baseline' in name:
        continue
    data=tmp[
        (tmp.top_k_preds_in_context > -1) &
        (tmp.neuron_subset_name.isin([name, name+'_baseline', 'clean']))
    ]
    plot_dir = f'{args.work_dir}/plots/ablations/{name}'
    if os.path.exists(plot_dir):
        #TODO continue?
        pass
    else:
        os.mkdir(plot_dir)
    plt.figure(figsize=(5,3))
    ax = sns.lineplot(data=data,
                    x="layer", y="top_k_preds_in_context",#we use zero-based layer indexing
                    hue="neuron_subset_name",
                    dashes=True,
                    linewidth=2,
                    markers=False,
                    palette=palette[:3],
                    )
    ax.set_xlabel("layer")
    ax.set_ylabel("attributes rate")
    plt.savefig(f'{plot_dir}/attributes.pdf', bbox_inches='tight')
    plt.close()
