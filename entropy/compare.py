#%%
from argparse import ArgumentParser
#from itertools import chain
import os

import torch

from ..plotting import aligned_histograms

#%%
def compute_data(data_path, metric, neuron_subset_name, intervention_type='zero_ablation'):
    #print('loading data...')
    baseline = torch.load(
        f'{data_path}/baseline/None_None/{metric}.pt',
        weights_only=True
    )
    ablated = torch.load(
        f'{data_path}/{neuron_subset_name}/{intervention_type}_None/{metric}.pt',
        weights_only=True
    )#sample pos
    #print('computing difference...')
    if metric=='scale':
        diff = baseline / ablated
    else:
        diff = baseline - ablated
    #print('flattening and removing zeros...')
    diff_flattened = diff.flatten()
    #remove zeros, corresponding to padding
    diff_nonzero = diff_flattened[diff_flattened.nonzero()].cpu().numpy()
    return diff_nonzero

def compare(args, metric, neuron_subset_names, intervention_type='zero_ablation', **kwargs):
    absrel = '/' if metric=='scale' else '-'
    data_path = f'{args.data_dir}/intervention_results/{args.model}/{args.dataset}'
    print('computing data...')
    diffs = {}
    baseline_names=[]
    for neuron_subset_name in neuron_subset_names:
        print(neuron_subset_name)
        diffs[neuron_subset_name] = compute_data(
            data_path, metric, neuron_subset_name, intervention_type
        )
        baseline_exists = os.path.exists(f'{data_path}/{neuron_subset_name}_baseline')
        if baseline_exists:
            baseline_name = neuron_subset_name+'_baseline'
            baseline_names.append(baseline_name)
            print(baseline_name)
            diffs[baseline_name] = compute_data(
                data_path, metric, baseline_name, intervention_type
            )
    # list_data = list(chain.from_iterable(
    #     (diffs[name], diffs[name+'_baseline']) for name in neuron_subset_names
    # ))
    list_data = list(diffs.values())
    #subtitles = list(chain.from_iterable((name, name+'_baseline') for name in neuron_subset_names))
    subtitles = list(k.replace('_', ' ', 1) for k in diffs)
    experiment_dir = f'{args.plot_dir}/{args.experiment_name}'
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)
    if args.log:
        kwargs["log"]=True
    aligned_histograms(
        list_data,
        subtitles=subtitles,
        savefile=f'{experiment_dir}/{metric}.pdf',
        suptitle = None,#f'{absrel} effect of neurons on {metric},\nas measured by {intervention_type}',
        xlabel=f'{metric}(clean) {absrel} {metric}(ablated)',
        ncols = 2,
        **kwargs
    )

#%%
if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir')
    parser.add_argument('--plot_dir')
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--model', default='allenai/OLMo-7B-0424-hf')
    parser.add_argument('--dataset', default='dolma_small')
    parser.add_argument('--metric', default='all')
    parser.add_argument('--intervention_type', default='zero_ablation')
    parser.add_argument(
        '--log', type=bool, default=True, help="logarithmic y-axis in the histograms"
    )#TODO different directory for log vs. linear histograms
    parser.add_argument('--neurons', nargs='+', default=['weakening'])
    args = parser.parse_args()
    if args.metric=='all':
        for metric in ['entropy', 'loss', 'rank', 'scale']:
            print(metric)
            compare(
                args, metric=metric, neuron_subset_names=args.neurons,
                intervention_type=args.intervention_type,
                #log=True
            )
    else:
        compare(
            args, metric=args.metric, neuron_subset_names=args.neurons,
            intervention_type=args.intervention_type,
            #log=True
        )
