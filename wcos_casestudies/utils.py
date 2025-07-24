# import random
# from functools import partial
# from tqdm import tqdm
# import matplotlib.pyplot as plt
import pandas as pd
import torch
import einops
# from transformer_lens.utils import to_numpy

CATEGORY_NAMES = [
    "enrichment", "atypical enrichment", "conditional enrichment", "atypical conditional enrichment",
    "proportional change", "atypical proportional change", "orthogonal output",
    "depletion", "atypical depletion", "conditional depletion", "atypical conditional depletion"
    ]

def topk_df(vec, model, emb=None, k=64, nonneg=True):
    """
    model used for to_string fct and emb
    """
    if emb is None:
        emb=model.W_U.detach()
    logits = torch.matmul(vec, emb)
    if not nonneg:
        logits_abs = torch.abs(logits)
        _, indices = torch.topk(logits_abs, k=k)
        logits = logits.cpu()
        values = [logits[i] for i in indices]
    else:
        values, indices = torch.topk(logits, k=k)
        values=values.cpu()
    str_tokens = [model.to_string(i) for i in indices]
    df = pd.DataFrame(values, index=str_tokens, columns=["dot product"])
    return df

def neuron_analysis(model, layer, neuron, emb=None, k=64, verbose=True):
    """
    emb: if None (default), unembedding matrix of model.
    Otherwise set explicit matrix of shape d_model, d_vocab.
    """
    out = model.W_out[layer,neuron,:].detach()
    lin = model.W_in[layer,:,neuron].detach()
    gate = model.W_gate[layer,:,neuron].detach()

    gatelin = cos(gate, lin)
    gateout = cos(gate, out)
    linout = cos(lin, out)

    out_pos = topk_df(out, model, emb=emb, k=k)
    out_neg = topk_df(-out, model, emb=emb, k=k)
    lin_pos = topk_df(lin, model, emb=emb, k=k)
    lin_neg = topk_df(-lin, model, emb=emb, k=k)
    gate_pos = topk_df(gate, model, emb=emb, k=k)
    gate_neg = topk_df(-gate, model, emb=emb, k=k)

    if verbose:
        print("gate vs. linear similarity:", gatelin)
        print("gate vs. out similarity:", gateout)
        print("lin vs. out similarity:", linout)
        print("================================")
        print("most similar tokens for w_out:")
        print(out_pos)
        print("================================")
        print("most similar tokens for -w_out:")
        print(out_neg)
        print("================================")
        print("most similar tokens for w_in:")
        print(lin_pos)
        print("================================")
        print("most similar tokens for -w_in:")
        print(lin_neg)
        print("================================")
        print("most similar tokens for w_gate:")
        print(gate_pos)
        print("================================")
        print("most similar tokens for -w_gate:")
        print(gate_neg)

    return gatelin, gateout, linout, out_pos, out_neg, lin_pos, lin_neg, gate_pos, gate_neg

def cos(v1,v2, pattern='... d, ... d -> ...'):
    v1 /= torch.linalg.vector_norm(v1, dim=-1)
    v2 /= torch.linalg.vector_norm(v2, dim=-1)
    dot = einops.einsum(v1, v2, pattern)
    return dot

def randomness_region(v1, v2, p=0.05):
    mismatched_cos = cos(v1, v2, 'l n1 d, l n2 d -> l (n1 n2)')
    low_quantile = torch.quantile(mismatched_cos, q=p/2, dim=-1)#l
    high_quantile = torch.quantile(mismatched_cos, q=1-(p/2), dim=-1)#l
    return torch.stack((low_quantile, high_quantile), dim=1)#l 2

def randomness_regions(mlp_weights, p=0.05):
    return {
        "gatelin": randomness_region(mlp_weights["W_gate"], mlp_weights["W_in"], p),
        "gateout": randomness_region(mlp_weights["W_gate"], mlp_weights["W_out"], p),
        "linout": randomness_region(mlp_weights["W_in"], mlp_weights["W_out"], p),
    }

# def wcos_ax(ax, gateout, linout, gatelin, s=1):
#     """
#     gateout, linout, gatelin must be on cpu
#     """
#     ax.set_xlim(-1.,1.)
#     ax.set_ylim(-1.,1.)
#     scatter = ax.scatter(gateout, linout, c=gatelin,  vmin=-1., vmax=1., linewidths=0, s=s)
#     return ax, scatter

# def wcos_plot(gateout, linout, gatelin, title=None, filename=None, s=1):
#     """
#     gateout etc are either 1d tensors (neuron) or 2d (layer,neuron), or lists of 1d tensors
#     """
#     if type(gateout)==list or (gateout.ndim==2 and gateout.shape[0]!=1):
#         if type(gateout)==list:
#             n=len(gateout)
#         else:
#             n = gateout.shape[0]
#         if n%4==0:
#             nrows=n//4
#             ncols=4
#         else:
#             nrows=n
#             ncols=1
#         fig, axs = plt.subplots(nrows,ncols, sharex=True, sharey=True)
#         axs_list = axs.ravel().tolist()
#         for layer in range(n):
#             ax=axs_list[layer]
#             ax, scatter = wcos_ax(ax, gateout[layer], linout[layer], gatelin[layer], s=s)
#             ax.set_title(f"Layer {layer}")
#     else:
#         fig, ax = plt.subplots()
#         axs_list = [ax]
#         ax, scatter = wcos_ax(ax, gateout, linout, gatelin, s=s)
#     fig.supxlabel("gate weights vs. output weights")
#     fig.supylabel("linear input weights vs. output weights")
#     fig.colorbar(scatter, ax=axs_list, label="gate vs. linear input weights")
#     if title is not None:
#         fig.suptitle(title)
#     if filename is not None:
#         fig.savefig(filename)
#     #plt.close(fig)

# def get_neuron_acts(model, text, layer, neuron_index, disentangle=True):
#     #https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Interactive_Neuroscope.ipynb
#     # Hacky way to get out state from a single hook -
#     # we have a single element list and edit that list within the hook.
#     cache = {}
#     def caching_hook(act, hook, name):
#         cache[name] = act[0, :, neuron_index]
#     fwd_hooks = [(f"blocks.{layer}.mlp.hook_post", partial(caching_hook, name="activation"))]
#     if disentangle:
#         fwd_hooks+=[
#             (f"blocks.{layer}.mlp.hook_pre", partial(caching_hook, name="gate")),
#             (f"blocks.{layer}.mlp.hook_pre_linear", partial(caching_hook, name="linear")),
#         ]
#     model.run_with_hooks(text, fwd_hooks=fwd_hooks)
#     return to_numpy(cache["activation"]), to_numpy(cache["gate"]), to_numpy(cache["linear"])

# def get_neuron_acts_on_dataset(model, layer, neuron, dataset, samples=None, seed=2512800, disentangle=True):
#     #https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Interactive_Neuroscope.ipynb
#     """deprecated, use get_multiple_neuron_acts instead"""
#     if samples is not None:
#         random.seed(seed)
#         sample = random.sample(range(len(dataset)), samples)
#     else:
#         sample = range(len(dataset))
#     all_acts=[]
#     if disentangle:
#         all_gate = []
#         all_lin = []
#     for sample_index, ds_index in tqdm(enumerate(sample)):
#         text = dataset[ds_index]['text']
#         #try:
#         if disentangle:
#             acts, gate, lin = get_neuron_acts(model, text, layer, neuron, disentangle=True)
#         else:
#             acts = get_neuron_acts(model, text, layer, neuron, disentangle=False)
#         # except:
#         #     acts = []
#         #     gate = []
#         #     lin = []
#         all_acts.append(acts)
#         if disentangle:
#             all_gate.append(gate)
#             all_lin.append(lin)
#     if disentangle:
#         return sample, all_acts, all_gate, all_lin
#     return sample, all_acts

# def get_multiple_neuron_acts(model, text, layer_neuron_dict, disentangle=True):
#     #https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Interactive_Neuroscope.ipynb
#     cache = {
#         "activation":{},
#         "gate":{},
#         "linear":{},
#     }
#     def caching_hook(act, hook, name, layer, neuron_list):
#         for ln in neuron_list:
#             cache[name][(layer,ln)] = to_numpy(act[0, :, ln])
#     fwd_hooks = [
#         (f"blocks.{layer}.mlp.hook_post", partial(caching_hook, name="activation", layer=layer, neuron_list=layer_neuron_dict[layer]))
#         for layer in layer_neuron_dict
#         ]
#     if disentangle:
#         fwd_hooks+=[
#             (f"blocks.{layer}.mlp.hook_pre", partial(caching_hook, name="gate", layer=layer, neuron_list=layer_neuron_dict[layer]))
#             for layer in layer_neuron_dict
#             ] + [
#                 (f"blocks.{layer}.mlp.hook_pre_linear", partial(caching_hook, name="linear", layer=layer, neuron_list=layer_neuron_dict[layer]))
#                 for layer in layer_neuron_dict
#                 ]
#     #print(fwd_hooks)
#     model.run_with_hooks(text, fwd_hooks=fwd_hooks)
#     #print(cache)
#     return cache

# def get_multiple_neuron_acts_on_dataset(model, neuron_list, dataset, samples=None, seed=2512800, disentangle=False):
#     #https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Interactive_Neuroscope.ipynb
#     if samples is not None:
#         random.seed(seed)
#         sample = random.sample(range(len(dataset)), samples)
#     else:
#         sample = range(len(dataset))

#     layer_neuron_dict = {l:[] for l in range(32)}#very hacky
#     for layer,neuron in neuron_list:
#         layer_neuron_dict[layer].append(neuron)
#     # for layer in layer_neuron_dict:
#     #     if not layer_neuron_dict[layer]:
#     #         del layer_neuron_dict[layer]

#     all_acts = {ln:[] for ln in neuron_list}
#     all_gate = {ln:[] for ln in neuron_list}
#     all_lin = {ln:[] for ln in neuron_list}
#     for sample_index, ds_index in tqdm(enumerate(sample)):
#         text = dataset[ds_index]['text']
#         cache = get_multiple_neuron_acts(model, text, layer_neuron_dict, disentangle=disentangle)
#         #cache should be a dict with "activation", "gate", "lin" as keys,
#         # and each of the values is a dict with keys (layer,neuron).
#         #assert (4,7696) in cache["activation"]
#         for ln in neuron_list:
#             all_acts[ln].append(cache["activation"][ln])
#             all_gate[ln].append(cache["gate"][ln])
#             all_lin[ln].append(cache["linear"][ln])
#         # del acts
#         # gc.collect()
#         # torch.cuda.empty_cache()
#     return sample, all_acts, all_gate, all_lin

# def find_examples(all_acts, cutoff=0.5):
#     """
#     Don't use this!
#     It typically finds way too many examples and then the files are impossible to open.
#     """
#     max_val = 0
#     min_val = 0
#     for acts in all_acts:
#         if any(acts > max_val):
#             max_val = max(acts)
#         if any(acts < min_val):
#             min_val = min(acts)
#     max_vals = []
#     min_vals = []
#     for sample_index, acts in enumerate(all_acts):
#         act_argmax = acts.argmax()
#         act_argmin = acts.argmin()
#         if acts[act_argmax] >= cutoff*max_val:
#             max_vals.append((sample_index, act_argmax))
#         if acts[act_argmin] <= cutoff*min_val:
#             min_vals.append((sample_index, act_argmin))
#     max_vals = [x for x in sorted(max_vals, key=lambda pair: -all_acts[pair[0]][pair[1]])]
#     min_vals = [x for x in sorted(min_vals, key=lambda pair: all_acts[pair[0]][pair[1]])]
#     return max_val, min_val, max_vals, min_vals

# STYLE_STRING = """<style> 
#     span.token {
#         border: 1px solid rgb(123, 123, 123)
#         } 
#     </style>"""

# def calculate_color(val, max_val, min_val, color="red"):
#     #https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Interactive_Neuroscope.ipynb
#     # Hacky code that takes in a value val in range [min_val, max_val], normalizes it to [0, 1]
#     # and returns a color which interpolates between slightly off-white and red (0 = white, 1 = red)
#     # We return a string of the form "rgb(240, 240, 240)" which is a color CSS knows
#     normalized_val = (val - min_val) / max_val
#     if color=="green":
#         return f"rgb({240*(1-normalized_val)}, 240, {240*(1-normalized_val)})"
#     elif color=="red":
#         return f"rgb(240, {240*(1-normalized_val)}, {240*(1-normalized_val)})"

# def neuron_vis(model, dataset, sample, sample_index, all_acts, layer, neuron_index, max_val=None, min_val=None, color="red"):
#     #https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Interactive_Neuroscope.ipynb
#     text = dataset[sample[sample_index]]['text']
#     str_tokens = model.to_str_tokens(text)
#     acts = all_acts[sample_index]

#     act_max = acts.max()
#     act_min = acts.min()
#     # Defaults to the max and min of the activations
#     if max_val is None:
#         max_val = act_max
#     if min_val is None:
#         min_val = act_min
#     # We want to make a list of HTML strings to concatenate into our final HTML string
#     # We first add the style to make each token element have a nice border
#     htmls = [STYLE_STRING]
#     # We then add some text to tell us what layer and neuron we're looking at -
#     # we're just dealing with strings and can use f-strings as normal
#     # h4 means "small heading"
#     htmls.append(f"<h4>Layer: <b>{layer}</b>. Neuron Index: <b>{neuron_index}</b></h4>")
#     # We then add a line telling us the limits of our range
#     htmls.append(
#         f"<h4>Max Range: <b>{max_val:.4f}</b>. Min Range: <b>{min_val:.4f}</b></h4>"
#     )
#     # If we added a custom range, print a line telling us the range of our activations too.
#     if act_max != max_val or act_min != min_val:
#         htmls.append(
#             f"<h4>Custom Range Set. Max Act: <b>{act_max:.4f}</b>. Min Act: <b>{act_min:.4f}</b></h4>"
#         )
#     for tok, act in zip(str_tokens, acts):
#         # A span is an HTML element that lets us style a part of a string
#         # (and remains on the same line by default)
#         # We set the background color of the span to be the color we calculated from the activation
#         # We set the contents of the span to be the token
#         htmls.append(
#             f"<span class='token' style='background-color:{calculate_color(act, max_val, min_val, color=color)};color:black' >{tok}</span>"
#         )

#     return "".join(htmls)

def count_categories(indices, gatelin, gateout, linout, threshold=.5):
    """
    Use this only for a small list of indices.
    To count categories on the whole model, use count_categories_all().
    If you have already done categories(),
    read everything off from the output tensor of that function.
    """
    d = 0
    cd = 0
    oo = 0
    pc = 0
    ce = 0
    e = 0
    other_d = 0
    other_cd = 0
    other_pc = 0
    other_ce = 0
    other_e = 0

    for ln in indices:
        l = ln[0]
        n = ln[1]
        if linout[l][n]<-threshold:
            if (gateout[l][n]<-threshold) or (gateout[l][n]>threshold):
                if torch.copysign(gatelin[l][n], gateout[l][n])<-threshold:
                    d+=1
                else:
                    other_d+=1
            else:
                if gatelin[l][n]>-threshold and gatelin[l][n]<threshold:
                    cd+=1
                else:
                    other_cd+=1
        elif linout[l][n]>threshold:
            if (gateout[l][n]<-threshold) or (gateout[l][n]>threshold):
                if torch.copysign(gatelin[l][n], gateout[l][n])>threshold:
                    e+=1
                else:
                    other_e+=1
            else:
                if gatelin[l][n]>-threshold and gatelin[l][n]<threshold:
                    ce+=1
                else:
                    other_ce+=1
        else:
            if gateout[l][n]>-threshold and gateout[l][n]<threshold:
                oo+=1
            else:
                if gatelin[l][n]>-threshold and gatelin[l][n]<threshold:
                    pc+=1
                else:
                    other_pc+=1
    return {"depletion": d,
            "atypical depletion": other_d,
            "conditional depletion": cd,
            "atypical conditional depletion": other_cd,
            "orthogonal output": oo,
            "proportional change": pc,
            "atypical proportional change": other_pc,
            "conditional enrichment": ce,
            "atypical conditional enrichment": other_ce,
            "enrichment": e,
            "atypical enrichment": other_e}

def category_lists(indices, gatelin, gateout, linout, threshold=.5, atypical=False):
    """
    Use this only for a small list of indices.
    To list all categories, use categories().
    If you have already done that,
    you can just read everything off from the output tensor of categories().
    """
    d = []
    cd = []
    oo = []
    pc = []
    ce = []
    e = []
    if atypical:
        other_d = []
        other_cd = []
        other_pc = []
        other_ce = []
        other_e = []

    for ln in indices:
        l = ln[0]
        n = ln[1]
        if linout[l][n]<-threshold:
            if (gateout[l][n]<-threshold) or (gateout[l][n]>threshold):
                if torch.copysign(gatelin[l][n], gateout[l][n])<-threshold:
                    d.append((l,n))
                elif atypical:
                    other_d.append((l,n))
            else:
                if gatelin[l][n]>-threshold and gatelin[l][n]<threshold:
                    cd.append((l,n))
                elif atypical:
                    other_cd.append((l,n))
        elif linout[l][n]>threshold:
            if (gateout[l][n]<-threshold) or (gateout[l][n]>threshold):
                if torch.copysign(gatelin[l][n], gateout[l][n])>threshold:
                    e.append((l,n))
                elif atypical:
                    other_e.append((l,n))
            else:
                if gatelin[l][n]>-threshold and gatelin[l][n]<threshold:
                    ce.append((l,n))
                elif atypical:
                    other_ce.append((l,n))
        else:
            if gateout[l][n]>-threshold and gateout[l][n]<threshold:
                oo.append((l,n))
            else:
                if gatelin[l][n]>-threshold and gatelin[l][n]<threshold:
                    pc.append((l,n))
                elif atypical:
                    other_pc.append((l,n))

    ans = {"depletion": d,
            "conditional depletion": cd,
            "orthogonal output": oo,
            "proportional change": pc,
            "conditional enrichment": ce,
            "enrichment": e,
    }
    if atypical:
        ans["atypical depletion"] = other_d
        ans["atypical conditional depletion"] = other_cd
        ans["atypical proportional change"] = other_pc
        ans["atypical conditional enrichment"] = other_ce
        ans["atypical enrichment"] = other_e

    return ans

def categories(gatelin, gateout, linout, threshold=.5):
    """Returns tensor of shape (layer, neuron)
    with integers indicating the category of the corresponding neuron,
    following CATEGORY_NAMES"""
    category_tensor = torch.full_like(gatelin, 6) #category_names[6]==orthogonal output
    category_tensor[(linout>threshold) &
            (torch.abs(gateout)>threshold) &
            (torch.copysign(gatelin, gateout)>threshold)] = 0 #enrichment
    category_tensor[(linout>threshold) &
            (torch.abs(gateout)>threshold) &
            (torch.copysign(gatelin, gateout)<=threshold)] = 1#'atypical enrichment'
    category_tensor[(linout>threshold) &
            (torch.abs(gateout)<=threshold) &
            (torch.abs(gatelin)<=threshold)] = 2#'conditional enrichment'
    category_tensor[(linout>threshold) &
            (torch.abs(gateout)<=threshold) &
            (torch.abs(gatelin)>threshold)] = 3#'atypical conditional enrichment'
    category_tensor[(torch.abs(linout)<=threshold) &
            (torch.abs(gateout)>threshold) &
            (torch.abs(gatelin)<=threshold)] = 4#'proportional change'
    category_tensor[(torch.abs(linout)<=threshold) &
            (torch.abs(gateout)>threshold) &
            (torch.abs(gatelin)>threshold)] = 5#'atypical proportional change'
    category_tensor[(linout<-threshold) &
            (torch.abs(gateout)>threshold) &
            (torch.copysign(gatelin, gateout)<-threshold)] = 7#'depletion'
    category_tensor[(linout<-threshold) &
            (torch.abs(gateout)>threshold) &
            (torch.copysign(gatelin, gateout)>=-threshold)] = 8#'atypical depletion'
    category_tensor[(linout<-threshold) &
            (torch.abs(gateout)<=threshold) &
            (torch.abs(gatelin)<=threshold)] = 9#'conditional depletion'
    category_tensor[(linout<-threshold) &
            (torch.abs(gateout)<=threshold) &
            (torch.abs(gatelin)>threshold)] = 10#'atypical conditional depletion'
    return category_tensor

def count_categories_all(category_tensor):
    """Output:
    dict with
    keys: strings corresponding to layers: '0', '1', etc.
    values: list of number of neurons (in that layer) for each class,
    where classes are ordered as in CATEGORY_NAMES
    """
    results = {str(l):[] for l in range(category_tensor.shape[0])}
    for i in range(len(CATEGORY_NAMES)):
        entry = torch.count_nonzero(category_tensor==i, dim=1)
        for l in range(category_tensor.shape[0]):
            results[str(l)].append(entry[l])
    return results

# def gather_class_changes(all_data, checkpoint_names=None):
#     """
#     returns: dict
#     with key (orig_class, new_class)
#     and value list of tuples (new_checkpoint, layer, neuron)
#     """
#     class_changes = {}
#     for i in range(11):
#         for j in range(11):
#             class_changes[(i,j)] = torch.empty((0,3)).cuda()
#     if checkpoint_names is None:
#         checkpoint_names = list(all_data.keys())
#     for checkpoint_nr, checkpoint_name in enumerate(checkpoint_names):
#         if checkpoint_nr == len(checkpoint_names)-1:
#             break
#         new_checkpoint_name = checkpoint_names[checkpoint_nr+1]
#         old_classes = all_data[checkpoint_name]['categories'].cuda()
#         new_classes = all_data[new_checkpoint_name]['categories'].cuda()
#         for t in class_changes:
#             indices = torch.nonzero((old_classes==t[0]) & (new_classes==t[1]))
#             #each row of indices is an index indicating: layer, neuron
#             indices_var = torch.cat(
#                 [
#                     torch.full(
#                         (indices.shape[0],1), checkpoint_nr+1
#                     ).cuda(),
#                     indices
#                 ],
#                 dim=1
#             )
#             #each row of indices_var indicates: new_checkpoint_nr, layer, neuron
#             class_changes[t] = torch.cat([class_changes[t], indices_var], dim=0)
#     for t in class_changes:
#         class_changes[t] = class_changes[t].cpu()
#     return class_changes
