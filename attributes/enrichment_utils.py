"""
Modified from Geva et al. 2023, Dissecting recall:
https://github.com/google-research/google-research/tree/master/dissecting_factual_predictions

Copyright 2023 Google LLC.
Licensed under the Apache License, Version 2.0 (the "License").
"""

from functools import partial
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch

from ..entropy.intervention import make_hooks

#%%
#from the utils.py of Geva et al, Dissecting Recall
def decode_tokens(tokenizer, token_array):
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode([t]) for t in token_array]
def find_token_range(tokenizer, token_array, substring):
    """Find the tokens corresponding to the given substring in token_array."""
    toks = decode_tokens(tokenizer, token_array)
    whole_string = "".join(toks)
    if substring in whole_string:
        char_loc = whole_string.index(substring)
    else:
        subject_var = substring[0].upper()+substring[1:]
        if subject_var in whole_string:
            char_loc = whole_string.index(subject_var)
        else:
            print(substring, subject_var)
            raise ValueError
    loc = 0
    tok_start, tok_end = None, None
    for i, t in enumerate(toks):
        loc += len(t)
        if tok_start is None and loc > char_loc:
            tok_start = i
        if tok_end is None and loc >= char_loc + len(substring):
            tok_end = i + 1
            break
    return (tok_start, tok_end)

#%%
#inspired by Gurnee et al., Universal Neurons
#new but boring:
# def make_layer_hooks(args, layer_list):
#     hooks = []
#     for layer in layer_list:
#         hooks += make_hooks(args, layer)
#     return hooks
def make_neuron_hooks(args, neuron_list):
    hooks = []
    if neuron_list is not None:
        for layer,neuron in neuron_list:
            hooks.extend(make_hooks(args, layer, neuron))
    return hooks

def caching_hook(res, hook, cache):
    cache[:,hook.layer(),:] = res
def make_caching_hooks(n_layers, cache):
    hooks = []
    for layer in range(n_layers):
        hooks.append((f'blocks.{layer}.hook_resid_post', partial(caching_hook, cache=cache)))
    return hooks


# %%
# inspired by the notebook,
# extended to arbitrary neuron subsets
def record_logitlens(args, knowns_df, model, neuron_subset=None, neuron_subset_name=None):
    """
    Inputs:
        knowns_df: a pandas dataframe of factual recall prompts
        model: a HookedTransformer
        neuron_subset: a list of neurons to zero-ablate,
        represented as (layer, neuron) tuples
    Output:
        a pandas dataframe with the logit lens computed
        at each layer
        for the subject of each prompt.
        Columns:
            example_index
            subject
            layer
            position
            neuron_subset_name
            topk_preds
    """
    records = []
    for row_i, row in tqdm(knowns_df.iterrows()):
        prompt = row.prompt
        subject = row.subject
        inp = model.to_tokens(prompt)[0]
        _, position = find_token_range(model.tokenizer, inp, subject)
        position -=1
        cache = torch.zeros(
            (position+1, model.cfg.n_layers, model.cfg.d_model),#pos layer d_model
            device='cuda'
        )
        with torch.no_grad():
            model.run_with_hooks(
                inp[:position+1],
                return_type=None,
                fwd_hooks=make_neuron_hooks(
                    args, neuron_subset
                ) + make_caching_hooks(
                        model.cfg.n_layers, cache
                    ),
            )

        for subject_repr_layer in range(model.cfg.n_layers):
            records.append({
                    "example_index": row_i,
                    "subject": subject,
                    "layer": subject_repr_layer,
                    "position": position,
                    "neuron_subset_name": neuron_subset_name,
                    "top_k_preds_str": logitlens(
                        model, cache[position,subject_repr_layer,:], args.topk
                    )
                })
    tmp = pd.DataFrame.from_records(records)
    return tmp

def logitlens(model, hs_, topk):
    projs_ = hs_.matmul(model.W_U).cpu().numpy()
    ind_ = np.argsort(-projs_)
    top_k_preds_ = model.to_str_tokens(ind_[:topk])
    return top_k_preds_
