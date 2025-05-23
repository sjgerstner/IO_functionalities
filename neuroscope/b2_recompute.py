import pickle

import torch
import einops

from transformers.activations import ACT2FN

def recompute_acts(model, layer, neuron, indices, save_path):
    actfn = ACT2FN[model.cfg.act_fn]

    ln_cache=[]
    for i in indices:
        with open(f"{save_path}/activation_cache/batch{i}.pickle", 'rb') as f:
            subcache = pickle.load(f)['ln_cache'] #batch pos layer d_model
            ln_cache.append(subcache[:,:,layer,:])
            #print(ln_cache[-1].shape)
    ln_cache = torch.cat(ln_cache).cuda()
    #print(ln_cache.shape) #should be sample pos d_model

    intermediate = {}

    intermediate['gate'] = actfn(
        einops.einsum(ln_cache,
                      model.blocks[layer].mlp.W_gate[:,neuron],
                      'sample pos d_model, d_model -> sample pos'
                      )
        )
    intermediate['in'] = einops.einsum(
        ln_cache,
        model.blocks[layer].mlp.W_in[:,neuron],
        'sample pos d_model, d_model -> sample pos'
        )
    intermediate['acts'] = intermediate['gate']*intermediate['in']
    # for key, value in intermediate.items():
    #     out_dict[key][layer,neuron] = value
    return intermediate
