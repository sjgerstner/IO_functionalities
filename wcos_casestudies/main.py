"""Main code for the IO functionalities paper"""

import argparse
import gc
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

import torch
import einops
from transformer_lens import HookedTransformer

import plotting
import utils

torch.set_grad_enabled(False)

DEVICE='cuda:0'

def _load_mlp_weights(model_name, cache_dir=None, checkpoint_value=None, refactor_glu=True):
    model = HookedTransformer.from_pretrained(
        model_name,
        checkpoint_value=checkpoint_value,
        cache_dir=cache_dir,
        local_files_only=True,
        device='cpu',
        refactor_glu=refactor_glu,
        ) #changed the transformer_lens code, don't disable fold_ln

    #new shape: layer neuron model_dim
    W_gate = einops.rearrange(model.W_gate.detach(), 'l d n -> l n d').to(DEVICE)
    W_in = einops.rearrange(model.W_in.detach(), 'l d n -> l n d').to(DEVICE)
    W_out = model.W_out.detach().to(DEVICE) #already has the shape we want

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {"W_gate":W_gate, "W_in":W_in, "W_out":W_out}

def cosines(mlp_weights):
    """Compute weight cosines within each neuron of the given model
    and return result as a dict of three tensors (gatelin, linout, gateout),
    each of which is of shape (layer neuron)."""
    gatelin = utils.cos(mlp_weights["W_gate"], mlp_weights["W_in"]).cpu() #don't overload the gpu
    linout = utils.cos(mlp_weights["W_in"], mlp_weights["W_out"]).cpu()
    gateout = utils.cos(mlp_weights["W_gate"], mlp_weights["W_out"]).cpu()
    data = {"gatelin":gatelin, "linout":linout, "gateout":gateout}
    return data

def analysis(model_name, args, cache_dir=None, checkpoint_value=None):
    """General function
    that computes weight cosines of the given model
    and then does the analyses specified in the args"""
    #path
    path = f"results/{model_name}"
    if args.refactor_glu:
        path += '/refactored'
    if checkpoint_value is not None:
        path += f"/checkpoints/{checkpoint_value}"
    if not os.path.exists(path):
        os.mkdir(path)

    #load and/or compute data and save it
    #cosines
    #if os.path.exists(f"{path}/data.pickle"):
    try:
        with open(f"{path}/data.pickle", 'rb') as f:
            data = pickle.load(f)
    #else:
    except RuntimeError as e:
        print(f"Ignoring error: {e}")
        mlp_weights = _load_mlp_weights(
            model_name,
            cache_dir=cache_dir, checkpoint_value=checkpoint_value,
            refactor_glu=args.refactor_glu,
        )
        data = cosines(mlp_weights)
    layers = data['gatelin'].shape[0]
    if args.randomness_regions and "randomness" not in data:
        data["randomness"] = utils.randomness_regions(mlp_weights)
    if checkpoint_value is not None:
        model_name=f"{model_name}/{checkpoint_value}"
    #categories and category statistics
    if args.categories and 'categories' not in data:
        data['categories'] = utils.categories(
            gatelin=data['gatelin'].to(DEVICE),
            gateout=data['gateout'].to(DEVICE),
            linout=data['linout'].to(DEVICE)
            ) #layer neuron
    if args.category_stats and 'category_stats' not in data:
        data['category_stats'] = utils.count_categories_all(data['categories'])
    if args.quartiles and 'quartiles' not in data:
        q = torch.tensor([0,.25,.5,.75,1])
        data['gatelin_quartiles'] = torch.quantile(data['gatelin'], q, dim=1)
        data['gateout_quartiles'] = torch.quantile(data['gateout'], q, dim=1)
        data['linout_quartiles'] = torch.quantile(data['linout'], q, dim=1)
    #save results:
    for key in data:
        data[key] = data[key].cpu()
    with open(f"{path}/data.pickle", "wb") as f:
        pickle.dump(data, f)

    #create various plots if requested and save them
    #fine-grained / cosines
    if args.plot_fine:# and not os.path.exists(f"{path}/fine.pdf"):
        ncols = 4
        fig, _ax = plotting.wcos_plot(
            range(layers),
            (int(np.ceil(layers/ncols)), ncols),
            data,
            model_name
            )
        fig.savefig(
            f"{path}/fine.pdf",
            bbox_inches='tight',
            #dpi=400
        )
        plt.close()
    #coarse-grained / category stats
    if args.plot_coarse:# and not os.path.exists(f"{path}/coarse.pdf"):
        fig, _ax = plotting.survey(data['category_stats'], model_name)
        fig.savefig(f"{path}/coarse.pdf", bbox_inches='tight')
        plt.close()
    #quartiles
    if args.plot_quartiles:# and not os.path.exists(f"{path}/quartiles.pdf")
        fig, _ax = plotting.plot_quartiles(data, model_name)
        fig.savefig(f"{path}/boxplot.pdf", bbox_inches='tight')
        plt.close()

    return data

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--refactor_glu',
        action='store_true',
        help='whether to refactor the weights such that cos(w_gate,w_in)>=0'
    )
    parser.add_argument(
        '--randomness',
        action='store_true',
        help="compute 95 percent randomness regions (95 percent of mismatched weight cosines are in this region)",
    )
    parser.add_argument(
        '--categories',
        action='store_true',
        help='categorize the neurons'
        )
    parser.add_argument(
        '--category_stats',
        action='store_true',
        help='compute statistics of IO classes by layer'
        )
    parser.add_argument(
        '--quartiles',
        action='store_true',
        help='compute quartiles of cosine similarities (by layer)'
        )
    parser.add_argument(
        '--plot_fine',
        action='store_true',
        help='create fine-grained plot'
        )
    parser.add_argument(
        '--plot_coarse',
        action='store_true',
        help='create coarse-grained plot (categories by layer)'
        )
    parser.add_argument(
        '--plot_quartiles',
        action='store_true',
        help='plot quartiles of cosine similarities by layer'
        )
    parser.add_argument(
        '--plot_all_medians',
        action='store_true',
        help="""
        make one plot
        with the median cos(w_in,w_out) similarities (y) across layers (x)
        of all models (one line per model)
        """,
    )
    parser.add_argument(
        "--model",
        nargs='+',
        default=["allenai/OLMo-7B-0424-hf"],
        help='one or several TransformerLens models, or "all"',
        )
    args = parser.parse_args()
    if args.model==['all']:
        model_list = [
            "allenai/OLMo-7B-0424-hf",
            "allenai/OLMo-1B-hf",
            "gemma-2-2b",
            "gemma-2-9b",
            "Llama-2-7b",
            "meta-llama/Llama-3.1-8B",
            "meta-llama/Llama-3.2-1B",
            "meta-llama/Llama-3.2-3B",
            "mistral-7b",
            "Qwen/Qwen2.5-0.5B",
            "Qwen/Qwen2.5-7B",
            "yi-6b",
        ]
    else:
        model_list = args.model
    if args.plot_all_medians:
        model_to_medians_dict = {}
    for model_name in model_list:
        data = analysis(model_name, args)
        if args.plot_all_medians:
            model_to_medians_dict[model_name] = data["linout_quartiles"][2,:].flatten().cpu()
    if args.plot_all_medians:
        fig, ax = plotting.plot_all_medians(model_to_medians_dict)
        fig.savefig('results/medians.pdf', bbox_inches='tight')
        plt.close()
