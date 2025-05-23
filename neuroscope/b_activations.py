"""
Compute max activations for all neurons on a given dataset.
The code was written with batch size 1 in mind.
Other batch sizes will almost certainly to bugs.
"""

from argparse import ArgumentParser
import os
import pickle
from tqdm import tqdm

import torch
import einops

from datasets import load_from_disk

from transformer_lens import HookedTransformer

HOOKS = ['ln2.hook_normalized', 'mlp.hook_post', 'mlp.hook_pre']
KEYS = [
        'ln_cache',
        'max_activations',
        'min_activations',
        'summary_mean',
        'summary_freq',
]
HEAP_KEYS = {
    'max_activations':'max',
    'min_activations':'min',
}
SUMMARY_KEYS = [
    'summary_mean',
    'summary_freq',
]
# SAVING_KEYS = [
#     'ln_cache',
#     'max_activations',
#     'argmax_activations',
#     'min_activations',
#     'argmin_activations',
# ]

def _move_to(dict_of_tensors, device):
    for key,value in dict_of_tensors.items():
        if torch.is_tensor(value):
            dict_of_tensors[key] = value.to(device)
        elif isinstance(value, dict):
            dict_of_tensors[key] = _move_to(value, device)
    return dict_of_tensors

def _get_reduce_and_arg(cache_item, reduction, k=1, to_device='cpu'):
    if reduction not in ('max', 'min', 'top', 'bottom'):
        raise NotImplementedError

    #... ? layer neuron -> ... k layer neuron
    myred = torch.topk(
        cache_item,
        dim=-3,
        k=k,
        largest=reduction in ('max','top'),
    )
    vi_dict = {'values': myred.values.to(to_device), 'indices':myred.indices.to(to_device)}
    if k==1:
        for key,tensor in vi_dict.items():
            vi_dict[key] = einops.reduce(tensor, '... 1 layer neuron -> ... layer neuron', reduction)
    return vi_dict

def _get_reduce(cache_item, reduction, arg=False, k=1, use_cuda=True, to_device='cpu'):
    if use_cuda and torch.cuda.is_available():
        cache_item = cache_item.cuda()

    if arg:
        return _get_reduce_and_arg(cache_item, reduction, k=k, to_device=to_device)
    return einops.reduce(
            cache_item,
            '... layer neuron -> layer neuron',
            reduction
            ).to(to_device)

def _get_all_neuron_acts(model, ids_and_mask, names_filter, max_seq_len=1024):
    #https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Interactive_Neuroscope.ipynb

    intermediate = {key:None for key in KEYS}

    batch_size = ids_and_mask['input_ids'].shape[0]
    seq_len = ids_and_mask['input_ids'].shape[1]

    _logits, raw_cache = model.run_with_cache(
        ids_and_mask['input_ids'],
        attention_mask=ids_and_mask['attention_mask'],
        names_filter=names_filter
        )
    #ActivationCache
    # with keys 'blocks.layer.mlp.hook_post' etc
    # and entries mostly with shape (batch pos neuron)

    mask = einops.rearrange(ids_and_mask['attention_mask'], 'batch pos -> batch pos 1 1').cuda()
    #batch pos neuron
    cache={}
    for hook in HOOKS:
        cache[hook] = torch.stack(
            [raw_cache[f'blocks.{layer}.{hook}'] for layer in range(model.cfg.n_layers)],
            dim=-2,#batch pos neuron/d_model -> batch pos layer neuron/d_model
        )
        cache[hook] *= mask
    del raw_cache

    #ln_cache: initialise with zeros (batch pos layer d_model)
    intermediate['ln_cache'] = torch.zeros(
        (batch_size, max_seq_len, model.cfg.n_layers, model.cfg.d_model)
        )
    #fill in
    intermediate['ln_cache'][:, :seq_len, :] = cache['ln2.hook_normalized'].cpu()

    for key,value in HEAP_KEYS.items():
        intermediate[key] = _get_reduce(
            cache['mlp.hook_post'],
            reduction=value,
            arg=True,
        )#batch pos layer neuron -> {'values': batch layer neuron, 'indices': batch layer neuron}

    #layer neuron
    intermediate['summary_mean'] = _get_reduce(cache['mlp.hook_post'], 'sum')
    intermediate['summary_freq'] = _get_reduce(cache['mlp.hook_pre']>0, 'sum')

    return intermediate

def get_all_neuron_acts_on_dataset(model, dataset, args, n_tokens=None, max_seq_len=1024, path=None):
    #https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Interactive_Neuroscope.ipynb

    batched_dataset = dataset.batch(
        batch_size=args.batch_size,
        drop_last_batch=False
        ) #preserves order

    names_filter = [
        f"blocks.{layer}.{hook}"
        for layer in range(model.cfg.n_layers)
        for hook in HOOKS
    ]

    if not os.path.exists(f'{path}/activation_cache'):
        os.mkdir(f'{path}/activation_cache')
    for i, batch in tqdm(enumerate(batched_dataset)):
        if i<args.resume_from or os.path.exists(f"{path}/activation_cache/batch{i}.pickle"):
            with open(f"{path}/activation_cache/batch{i}.pickle", 'rb') as f:
                intermediate = _move_to(pickle.load(f), device='cuda')
        else:
            intermediate = _get_all_neuron_acts(model, batch, names_filter, max_seq_len)
            with open(f"{path}/activation_cache/batch{i}.pickle", 'wb') as f:
                pickle.dump(_move_to(intermediate, 'cpu'), f)
        if i==0:
            out_dict = {key:intermediate[key] for key in SUMMARY_KEYS}
            for key in HEAP_KEYS:
                out_dict[key] = {
                    'values':intermediate[key]['values'],
                    'indices':torch.full_like(intermediate[key]['values'], 0)
                }
        else:
            for key in KEYS:
                if key in SUMMARY_KEYS:
                    out_dict[key] = _get_reduce(
                        torch.stack([out_dict[key], intermediate[key]]),
                        'sum'
                        )#batch layer neuron -> layer neuron
                elif key in HEAP_KEYS:
                    out_dict[key] = {
                        'values': torch.cat(
                            [
                                out_dict[key]['values'],
                                intermediate[key]['values']
                                ]
                            ),
                        'indices':torch.cat(
                            [
                                out_dict[key]['indices'],
                                torch.full_like(intermediate[key]['values'], i)
                                ]
                            )
                    }#both entries: sample layer neuron
                    if i>=args.examples_per_neuron:#running topk computation
                        #print(out_dict[key]['values'].shape) #should be: k layer neuron
                        vi = _get_reduce(
                            out_dict[key]['values'],
                            HEAP_KEYS[key],
                            arg=True,
                            k=args.examples_per_neuron,
                            )#k+1 layer neuron -> k layer neuron
                        out_dict[key]['values'] = vi['values']
                        out_dict[key]['indices'] = torch.gather(out_dict[key]['indices'], dim=0, index=vi['indices'])
                        #original dataset indices!
                        #I want:
                        #output[i,layer,neuron] = out_dict[key]['indices'][vi['indices'][i,layer,neuron],layer,neuron]
                        #hence
                        #output = torch.gather(out_dict[key]['indices'], dim=0, index=vi['indices'])
                else:
                    del intermediate[key]

    for key in SUMMARY_KEYS:
        out_dict[key] = out_dict[key].to(torch.float) / float(n_tokens)

    return out_dict

def topk_indices(maxact, k=16, largest=True, use_cuda=True):
    if use_cuda and torch.cuda.is_available():
        maxact = maxact.cuda()
    _values, indices = torch.topk(maxact, k=k, dim=0, largest=largest)
    #sample layer neuron -> k layer neuron, entries are indices along sample dimension
    return indices

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset', default='dolma_small')
    parser.add_argument('--model', default='allenai/OLMo-1B-hf')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--examples_per_neuron', default=16, type=int)
    parser.add_argument('--resume_from', default=0)
    parser.add_argument('--datadir', default=None)
    parser.add_argument('--save_to', default=None)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    if args.save_to:
        RUN_CODE = args.save_to
    elif args.test:
        RUN_CODE = "test"
    else:
        RUN_CODE = f"{args.model.split('/')[-1]}_{args.dataset.split('-')[0]}"
    #OLMO-1B-hf_dolma-v1_7-3B

    SAVE_PATH = f"{args.datadir}/{RUN_CODE}"
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)

    torch.set_grad_enabled(False)

    model = HookedTransformer.from_pretrained(args.model)

    dataset = load_from_disk(f'{args.datadir}/{args.dataset}')
    if args.test:
        dataset = dataset.select(range(2))
    n_tokens = sum(len(row) for row in dataset['input_ids'])
    max_seq_len = max(len(row) for row in dataset['input_ids'])

    print('computing activations...')
    if not os.path.exists(f'{SAVE_PATH}/summary.pickle'):
        out_dict = get_all_neuron_acts_on_dataset(model,
                                                dataset,
                                                args=args,
                                                max_seq_len=max_seq_len,
                                                n_tokens=n_tokens,
                                                path=SAVE_PATH,
                                                )
        with open(f'{SAVE_PATH}/summary.pickle', 'wb') as f:
            pickle.dump(_move_to(out_dict, 'cpu'), f)
    print('done!')
