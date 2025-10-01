"""
Largely copied from Gurnee et al. 2024, Universal neurons

MIT License

Copyright (c) 2023 Wes Gurnee

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import os
#import time
import tqdm
import torch
#import einops
import datasets
import argparse
# import numpy as np
# import pandas as pd
from functools import partial
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import DataCollatorWithPadding
from transformer_lens import HookedTransformer
from transformer_lens.utils import lm_cross_entropy_loss

from .utils import get_model_family
from .activations import get_correct_token_rank
from .intervention import (
    zero_ablation_hook,
    threshold_ablation_hook,
    relu_ablation_hook,
    fixed_activation_hook,
    #quantize_neurons
    #masked_zero_ablation_hook,
)

def multiply_activation_hook(activations, hook, neuron, multiplier=1):
    activations[:, :, neuron] = activations[:, :, neuron] * multiplier
    return activations

def save_layer_norm_scale_hook(activations, hook):
    hook.ctx['activation'] = activations.detach().cpu()


def store_conditioning_hook(activation, hook, conditioning_values, neuron=None, sign=1):
    """Store the conditioning activation"""
    if neuron is not None:
        conditioning_values[hook.name] = activation[:,:,neuron].clone()
        if sign<0:
            conditioning_values[hook.name] *=-1
    else:
        conditioning_values[hook.name] = activation.clone()


def conditional_ablation_hook(activation, hook, args, conditioning_value, neuron=None):
    """Ablate based on conditioning activation"""
    hook_loc = '.'.join(hook.name.split('.')[:-1])
    gate_loc = hook_loc+'.hook_pre'
    post_loc = hook_loc+'.hook_post'
    condition = None
    if gate_loc in conditioning_value:
        if args.gate=='+':
            condition = conditioning_value[gate_loc]>0
        elif args.gate=='-':
            condition = conditioning_value[gate_loc]<0
        else:
            raise NotImplementedError("argument 'gate' has to be + or -")
    if post_loc in conditioning_value:
        if condition is None:
            condition = torch.ones_like(conditioning_value[post_loc])
        if args.post=='+':
            condition = torch.logical_and(condition, conditioning_value[post_loc]>0)
        elif args.post=='-':
            condition = torch.logical_and(condition, conditioning_value[post_loc]<0)
        else:
            raise NotImplementedError("argument 'post' has to be + or -")
    activation = ablation_hook(activation, hook, args, neuron=neuron, mask=condition)
    return activation

def ablation_hook(activation, hook, args, neuron=None, mask=None):
    """Branch to the right intervention type based on args.intervention_type"""
    if mask is None:
        mask=torch.ones(activation.shape[:-1], dtype=bool)
    if args.intervention_type == 'zero_ablation':
        # if mask is not None:
        #     activation = masked_zero_ablation_hook(activation,hook,neuron,mask)
        #else:
        activation = zero_ablation_hook(activation, hook, neuron=neuron, mask=mask)
    #TODO implement mask for the other intervention types
    elif args.intervention_type == 'threshold_ablation':
        activation = threshold_ablation_hook(
            activation, hook, neuron=neuron,
            threshold=args.intervention_param
        )
    elif args.intervention_type == 'fixed_activation':
        activation = fixed_activation_hook(
            activation, hook,
            neuron=neuron,
            fixed_act=args.intervention_param
        )
    elif args.intervention_type == 'relu_ablation':
        activation = relu_ablation_hook(activation, hook, neuron=neuron)

    elif args.intervention_type == 'multiply_activation':
        activation = multiply_activation_hook(
            activation, hook,
            neuron=neuron,
            multiplier=args.intervention_param)
    else:
        raise ValueError(
            f'Unknown intervention type: {args.intervention_type}')
    return activation

def make_hooks(args, layer, neuron, conditioning_value=None, sign=1):
    out = []

    if args.gate is not None:
        hook_loc = f'blocks.{layer}.mlp.hook_pre'
        hook_fn = partial(
            store_conditioning_hook,
            conditioning_values=conditioning_value, neuron=neuron
        )
        out.append((hook_loc, hook_fn))
    if args.post is not None:
        hook_loc = f'blocks.{layer}.mlp.hook_post'
        hook_fn = partial(
            store_conditioning_hook,
            conditioning_values=conditioning_value, neuron=neuron,
            sign=sign,
        )
        out.append((hook_loc, hook_fn))

    # if args.intervention_type == 'zero_ablation':
    #     hook_fn = partial(zero_ablation_hook, neuron=neuron, args=args)
    # elif args.intervention_type == 'threshold_ablation':
    #     hook_fn = partial(
    #         threshold_ablation_hook,
    #         neuron=neuron,
    #         threshold=args.intervention_param)
    # elif args.intervention_type == 'fixed_activation':
    #     hook_fn = partial(
    #         fixed_activation_hook,
    #         neuron=neuron,
    #         fixed_act=args.intervention_param)
    # elif args.intervention_type == 'relu_ablation':
    #     hook_fn = partial(relu_ablation_hook, neuron=neuron)

    # elif args.intervention_type == 'multiply_activation':
    #     hook_fn = partial(
    #         multiply_activation_hook,
    #         neuron=neuron,
    #         multiplier=args.intervention_param)
    # else:
    #     raise ValueError(
    #         f'Unknown intervention type: {args.intervention_type}')
    hook_fn = partial(
        conditional_ablation_hook,
        conditioning_value=conditioning_value,
        args=args, neuron=neuron)

    hook_loc = f'blocks.{layer}.{args.activation_location}'
    out.append((hook_loc, hook_fn))

    return out


def run_intervention_experiment(
    args,
    model,
    dataset,
    device,
    neuron_subset=None,
    neuron_subset_name=None,
    save_path=None,
    ):

    if neuron_subset is None:
        neuron_subset = args.neuron_subset

    conditioning_values = {}
    hooks = []
    for lix, nix in neuron_subset:
        conditioning_values[(lix,nix)]={}
        hooks += make_hooks(
            args,
            lix, nix,
            conditioning_values[(lix,nix)],
            sign=(model.W_gate[lix,:,nix]@model.W_in[lix,:,nix]).item()
        )

    hooks.append(('ln_final.hook_scale', save_layer_norm_scale_hook))

    #n, d = dataset['tokens'].shape
    n = dataset.num_rows
    d = max(len(example["input_ids"]) for example in dataset)
    #print(d)
    loss_tensor = torch.zeros(n, d, dtype=torch.float16)
    entropy_tensor = torch.zeros(n, d, dtype=torch.float16)
    rank_tensor = torch.zeros(n, d, dtype=torch.int32)
    scale_tensor = torch.zeros(n, d, dtype=torch.float16)

    data_collator = DataCollatorWithPadding(
        tokenizer=model.tokenizer,
        # padding='max_length',
        # max_length=d,
    )
    dataloader = DataLoader(
        dataset,#dataset['tokens'],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    offset = 0
    for step, batch in enumerate(tqdm.tqdm(dataloader)):
        batch = batch.to(device)
        logits = model.run_with_hooks(
            input=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            fwd_hooks=hooks
        )
        #print(logits.shape)
        bs, local_seq_len = batch["input_ids"].shape
        token_loss = lm_cross_entropy_loss(
            logits,
            batch["input_ids"],
            batch["attention_mask"],
            per_token=True
        ).cpu()
        probs = F.softmax(logits, dim=-1)
        entropies = (
            - torch.sum(probs * torch.log(probs + 1e-8), dim=-1)*batch["attention_mask"]
        ).cpu()
        token_ranks = (
            get_correct_token_rank(logits, batch["input_ids"])*batch["attention_mask"][:,:-1]
        ).cpu()

        loss_tensor[offset:offset+bs, :local_seq_len-1] = token_loss
        entropy_tensor[offset:offset+bs, :local_seq_len] = entropies
        rank_tensor[offset:offset+bs, :local_seq_len-1] = token_ranks

        scale = model.hook_dict['ln_final.hook_scale'].ctx['activation'].squeeze()
        scale_tensor[offset:offset+bs, :local_seq_len] = scale

        offset += bs

        model.reset_hooks()
    if not save_path:
        if not neuron_subset_name:
            neuron_subset_name = '_'.join([f'{l}.{n}' for l, n in neuron_subset])
        save_path = os.path.join(
            args.output_dir,
            args.model,
            args.token_dataset,
            neuron_subset_name,
            str(args.intervention_type)+'_'+str(args.intervention_param),
        )
    os.makedirs(save_path, exist_ok=True)
    torch.save(loss_tensor, os.path.join(save_path, f'loss.pt'))
    torch.save(entropy_tensor, os.path.join(
        save_path, f'entropy.pt'))
    torch.save(rank_tensor, os.path.join(save_path, f'rank.pt'))
    torch.save(scale_tensor, os.path.join(save_path, f'scale.pt'))


def parse_neuron_str(neuron_str: str):
    neurons = []
    for group in neuron_str.split(','):
        lix, nix = group.split('.')
        neurons.append((int(lix), int(nix)))
    return neurons


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # general arguments
    parser.add_argument(
        '--model', default='stanford-gpt2-small-a',
        help='Name of model from TransformerLens')
    parser.add_argument(
        '--token_dataset',
        help='Name of cached feature dataset')
    parser.add_argument(
        '--activation_location', default='mlp.hook_post',
        help='Model component to save')

    # activation processing/subsetting arguments
    parser.add_argument(
        '--batch_size', default=32, type=int)
    parser.add_argument(
        '--device', default=torch.device('cuda' if torch.cuda.is_available() else (
            'mps' if torch.backends.mps.is_available() else 'cpu')), type=str,
    )

    parser.add_argument(
        '--neuron_subset', type=parse_neuron_str, default=None,
        help='list of neurons')

    parser.add_argument(
        '--intervention_type', type=str, default='fixed_activation',
        help='Type of intervention to perform')
    parser.add_argument(
        '--intervention_param', type=float, default=None,
        help='Parameter for intervention type (eg, threshold or fixed activation)')

    # saving arguments
    parser.add_argument(
        '--save_precision', default=16, type=int)
    parser.add_argument(
        '--output_dir', default='intervention_results')

    args = parser.parse_args()

    device = args.device

    model = HookedTransformer.from_pretrained(args.model, device='cpu')
    model.to(device)
    model.eval()
    torch.set_grad_enabled(False)
    model_family = get_model_family(args.model)

    tokenized_dataset = datasets.load_from_disk(
        os.path.join(
            os.getenv('DATASET_DIR', 'token_datasets'),
            model_family,
            args.token_dataset
        )
    )

    run_intervention_experiment(
        args, model, tokenized_dataset, device)
