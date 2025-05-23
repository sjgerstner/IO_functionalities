from argparse import ArgumentParser
import os
import pickle
from tqdm import tqdm

import torch
from datasets import load_from_disk
from transformer_lens import HookedTransformer

from b_activations import _move_to
from b2_recompute import recompute_acts
from c_neuron_vis import neuron_vis_full

parser = ArgumentParser()
parser.add_argument('--dataset', default='dolma_small')
parser.add_argument('--model', default='allenai/OLMo-1B-hf')
parser.add_argument('--datadir', default=None)
parser.add_argument('--save_to', default=None)
parser.add_argument('--neurons',
    nargs='+',
    default=[],
    help='one or several neurons denoted as layer.neuron, or "all"',
)
parser.add_argument('--test', action='store_true')
args = parser.parse_args()

print("preparation...")
if args.save_to:
    RUN_CODE = args.save_to
elif args.test:
    RUN_CODE = "test"
else:
    RUN_CODE = f"{args.model.split('/')[-1]}_{args.dataset.split('-')[0]}"
#OLMO-1B-hf_dolma-v1_7-3B
#the id of the b_activations.py run

SAVE_PATH = f"{args.datadir}/{RUN_CODE}"
TITLE = f"<h1>Model: <b>{args.model}</b></h1>\n"

torch.set_grad_enabled(False)

model = HookedTransformer.from_pretrained(args.model)
tokenizer = model.tokenizer

with open(f'{SAVE_PATH}/summary.pickle', 'rb') as f:
    summary_dict = _move_to(pickle.load(f), 'cuda')
    print(f"summary_dict: {summary_dict.keys()}")

TOPK = summary_dict['max_activations']['indices'].shape[0]#topk layer neuron
if args.neurons=='all':
    layer_neuron_list = [range(model.cfg.d_mlp) for _layer in range(model.cfg.n_layers)]
elif args.neurons:
    layer_neuron_list = [[] for layer in range(model.cfg.n_layers)]
    for ln_str in args.neurons:
        layer, neuron = tuple(int(n) for n in ln_str.split('.'))
        layer_neuron_list[layer].append(neuron)
elif args.test:
    layer_neuron_list = [[0]]

summary_dict['max_activations']['indices'] = summary_dict['max_activations']['indices'].to(torch.int)
summary_dict['min_activations']['indices'] = summary_dict['min_activations']['indices'].to(torch.int)

maxmin_indices = torch.cat(
    [summary_dict['max_activations']['indices'], summary_dict['min_activations']['indices']]
)#.to(torch.int)

dataset = load_from_disk(f'{args.datadir}/{args.dataset}')

for layer,neuron_list in enumerate(layer_neuron_list):
    print(f'processing layer {layer}...')
    if not os.path.exists(f"{args.datadir}/{RUN_CODE}/L{layer}"):
        os.mkdir(f"{args.datadir}/{RUN_CODE}/L{layer}")
    for neuron in tqdm(neuron_list):
        neuron_dir = f"{args.datadir}/{RUN_CODE}/L{layer}/N{neuron}"
        if not os.path.exists(neuron_dir):
            os.mkdir(neuron_dir)
        #recomputing neuron activations on max and min examples
        if not os.path.exists(f'{neuron_dir}/activations.pt'):
            dict_all = recompute_acts(
                model,
                layer, neuron,
                maxmin_indices[:,layer,neuron],
                # out_dict=dict_all,#dict_all is just updated
                save_path=SAVE_PATH,
            )
            torch.save(dict_all, f'{neuron_dir}/activations.pt')
        else:
            dict_all = torch.load(f'{neuron_dir}/activations.pt')
        #visualisation
        neuron_data = {
            'max_indices':maxmin_indices[:TOPK, layer, neuron],
            'min_indices':maxmin_indices[TOPK:, layer, neuron],
            'max_acts':dict_all['acts'][:TOPK],
            'min_acts':dict_all['acts'][TOPK:],
            'max_val':summary_dict['max_activations']['values'][0,layer,neuron],
            'min_val':summary_dict['min_activations']['values'][0,layer,neuron],
            'avg_val':summary_dict['summary_mean'][layer,neuron],
            'act_freq':summary_dict['summary_freq'][layer,neuron],
            'argmax_tokens':summary_dict['max_activations']['indices'][:,layer,neuron],
            'argmin_tokens':summary_dict['min_activations']['indices'][:,layer,neuron],
        }
        # We add some text to tell us what layer and neuron we're looking at
        heading = f"<h2>Layer: <b>{layer}</b>. Neuron Index: <b>{neuron}</b></h2>\n"
        HTML = TITLE + heading + neuron_vis_full(
                neuron_data=neuron_data,
                dataset=dataset,
                tokenizer=tokenizer,
        )
        with open(f'{neuron_dir}/vis.html', 'w', encoding="utf-8") as f:
            f.write(HTML)
print('done!')
