# %% [markdown]
# # Code for sections "IO classes vs. functional roles" and the parameter-based analyses of "Case studies"

# %% [markdown]
# I did not actually run this particular script, so please be patient with any bugs.
# If the GPU is overloaded at some point, you can move tensors to the CPU and/or delete variables
# (then run gc.collect() and torch.cuda.empty_cache()).
# In particular you can delete those variables that have been pickled and unpickle them at need.

# %% [markdown]
# ## Setup

# %%
import gc
import pickle

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name

import utils

# %%
torch.no_grad()
pd.set_option('display.max_rows', None)

# %%
MODEL_NAME = "allenai/OLMo-7B-0424-hf" #Not supported by the original TransformerLens!

# %%
model = HookedTransformer.from_pretrained(MODEL_NAME, device='cuda')

# %%
W_out = model.W_out.detach()
W_U = model.W_U.detach()

#%%
#load the data from running main.py
with open(f'results/{MODEL_NAME}/data.pickle', 'rb') as f:
    data = pickle.load(f)
gatelin = data['gatelin'].cuda()
gateout = data['gateout'].cuda()
linout = data['linout'].cuda()

# %%
WUWout = torch.einsum("l n d, d v -> l n v", W_out, W_U)
print(WUWout.shape)

# %%
norm_out = torch.linalg.vector_norm(W_out, dim=-1, keepdim=True)#l n 1
print(norm_out.shape)
norm_U = torch.linalg.vector_norm(W_U, dim=0, keepdim=True)#1 v
print(norm_U.shape)

# %%
norm_U = torch.unsqueeze(norm_U, dim=0)

# %%
WUWout /= norm_out
WUWout /= norm_U
print(WUWout.shape)

# %%
with open('cosWUWout.pickle', 'wb') as f:
    pickle.dump(WUWout, f)

# %% [markdown]
# #### Partition neurons

# %%
variances = torch.var(WUWout, dim=2).cpu() #layer neuron

# %%
#top 1000 PARTITION neurons
p_variances, p_indices = torch.topk(variances.flatten(), k=1000)
print('lowest variance considered:', p_variances[-1])

p_indices_layer = p_indices // 11008
p_indices_neuron = p_indices % 11008
p_indices_readable = torch.stack(
    [[p_indices_layer[i], p_indices_neuron[i]] for i in range(len(p_indices_layer))]
)

# %%
with open('partition.pickle', 'wb') as f:
    pickle.dump(p_indices_readable, f)

# %% [markdown]
# #### Prediction / suppression neurons

# %%
means = torch.mean(WUWout, dim=2).cpu() #layer neuron
diffs = WUWout.cpu() - torch.unsqueeze(means,2)
stds = torch.sqrt(variances)
zscores = diffs / torch.unsqueeze(stds,2)

# %%
skews = torch.mean(torch.pow(zscores, 3.0), dim=2)

# %%
kurtoses = torch.mean(torch.pow(zscores, 4.0), dim=2) - 3.0
#note the -3: shows excess kurtosis ("raw" kurtosis of Gaussian is 3, excess kurtosis is 0)

# %%
max_part_kurt = 0
for l,n in zip(p_indices_layer, p_indices_neuron):
    if kurtoses[l,n]>max_part_kurt:
        max_part_kurt = kurtoses[l,n]
print(max_part_kurt)

# %%
ps = kurtoses>max_part_kurt
ps_indices = ps.nonzero()
print(ps_indices)

# %%
# The following is necessary to (approximately) distinguish prediction from suppression in a gated activation function
pred_indices = []
supp_indices = []
for ln in ps_indices:
    #print(ln)
    if torch.sign(gatelin[ln[0], ln[1]])*torch.sign(skews[ln[0], ln[1]])>0:
        pred_indices.append(ln)
    else:
        supp_indices.append(ln)

# %%
pred_indices = torch.stack(pred_indices)
supp_indices = torch.stack(supp_indices)

# %%
with open('pred-supp.pickle', 'wb') as f:
    pickle.dump({"prediction":pred_indices, "suppression":supp_indices}, f)

# %% [markdown]
# #### Entropy neurons

# %%
norms = torch.linalg.vector_norm(W_out[-1,:,:], dim=-1)

# %%
U,S,V = torch.svd(W_U)

# %%
U_null = U[:,4056:]

# %%
null_norm_ratios = torch.linalg.vector_norm(
    torch.einsum("... n d, d v -> ... n v", W_out, U_null), dim=-1
) / norms

# %%
fig, ax = plt.subplots()
ax.boxplot(null_norm_ratios)
fig.show()

# %% [markdown]
# The above figure shows that two neurons stand out.
# The following code prints their null norm ratios (ratios) and their neuron indices within layer 31 (indices).

# %%
ratios, entropy_indices = torch.topk(null_norm_ratios, k=2)
print(ratios, entropy_indices)

# %% [markdown]
# #### Attention (de)activation neurons

# %%
#from Gurnee et al.'s code
_, BOS_cache = model.run_with_cache(model.to_tokens(""),
    names_filter=[get_act_name('k', i) for i in range(model.cfg.n_layers)]
)

# %%
print(BOS_cache['blocks.0.attn.hook_k'].shape)

# %% [markdown]
# (batch pos n_heads d_head)

# %%
W_out_normalised = W_out / torch.linalg.vector_norm(W_out, dim=-1, keepdim=True)

# %% [markdown]
# It may be better to normalise the qk vector too, so that results are comparable across heads.

# %%
Qk_normalised = []
scores_normalised = []
for attn_layer in range(1,32):
    qk = torch.einsum(
        'h d, h D d -> h D',
        BOS_cache[f'blocks.{attn_layer}.attn.hook_k'].squeeze(),
        model.W_Q.detach()[attn_layer,:,:,:]
    )
    qk_normalised = qk / torch.linalg.vector_norm(qk, dim=1, keepdim=True)
    Qk_normalised.append(qk_normalised)
    scores_normalised.append(
        torch.einsum('l n D, h D -> l n h', W_out_normalised[:attn_layer,:,:], qk_normalised)
    )

# %%
for s in scores_normalised:
    print(s.shape)

# %%
scores_normalised_tensor = torch.zeros((31, 31, 11008, 32)) #attn_layer, neuron_layer, neuron, head
for attn_layer, s in enumerate(scores_normalised):
    scores_normalised_tensor[attn_layer, :attn_layer+1, :, :] = s #neuron_layer, neuron, head

print(scores_normalised_tensor.shape)

# %% [markdown]
# attn_layer (from 1 to 31), mlp_layer (from 0 to 30), neuron, head

# %% [markdown]
# In the following we focus only on
# attention (de)activation neurons that (seem to) affect the next layer.

# %%
scores_diagonal = torch.diagonal(scores_normalised_tensor)
print(scores_diagonal.shape)
#neuron, head in following layer, mlp layer

# %%
cutoff = np.sqrt(.5)

# %%
special_diagonal = (torch.abs(scores_diagonal)>cutoff)
n_special_diagonal = torch.count_nonzero(special_diagonal, dim=1).T #number of heads a given neuron (de)activates
print(n_special_diagonal.shape)
print(n_special_diagonal.count_nonzero())
ad_indices = torch.nonzero(n_special_diagonal)
print(ad_indices) #indices of attention (de)activation neurons
for i in ad_indices:
    print(n_special_diagonal[i[0],i[1]]) #number of affected heads for each of them

# %%
for i in ad_indices:
    print(scores_diagonal[i[1], :, i[0]])

# %%
for i in ad_indices:
    print(torch.max(torch.abs(scores_diagonal[i[1], :, i[0]])))

# %% [markdown]
# For now we have the problem that some of these neurons
# are also prediction, suppression or partition neurons.

# %%
for i in ad_indices:
    if f"{i[0]}.{i[1]}" in p_indices_readable:
        print(f"{i} is a partition neuron.")
    elif i in pred_indices:
        print(f"{i} is a prediction neuron.")
    elif i in supp_indices:
        print(f"{i} is a suppression neuron.")
    else:
        print(f"{i} is free!")

# %% [markdown]
# We count only the first four as attention (de)activation.
# 
# We still need to find out if it's activation or deactivation:

# %%
for i in ad_indices[:4,:]:
    print("======")
    print("neuron", i)
    if gatelin[i[0],i[1]] * scores_diagonal[i[1],i[0], torch.argmax(torch.abs(scores_diagonal[i[1],i[0],:]))] > 0:
        print("deactivation")
    else:
        print("activation")

# %%
scores_to_pickle = torch.stack([scores_diagonal[i[1],:,i[0]] for i in ad_indices])
with open('attention.pickle', 'wb') as f:
    pickle.dump([ad_indices, scores_to_pickle], f)

# %% [markdown]
# ### Comparing with our classification

# %%
print("Partition:")
part_io = utils.count_categories(p_indices_readable, gatelin, gateout, linout)
print(part_io)

# %%
print("Prediction:")
pred_io = utils.count_categories(pred_indices, gatelin, gateout, linout)
print(pred_io)

# %%
print("Suppression:")
supp_io = utils.count_categories(supp_indices, gatelin, gateout, linout)
print(supp_io)

# %%
print("Entropy:")
ent_io = utils.count_categories(entropy_indices, gatelin, gateout, linout)
print(ent_io)

# %%
print("Attention deactivation:")
ad_io = utils.count_categories(ad_indices[:4,:], gatelin, gateout, linout)
print(ad_io)

# %%
with open("contingency.pickle", 'wb') as f:
    pickle.dump(
        {
            "partition": part_io,
            "prediction": pred_io,
            "suppression": supp_io,
            "entropy": ent_io,
            "attention deactivation":ad_io
        },
        f
    )
# %% [markdown]
# ## Case studies

# %% [markdown]
# ### Finding neurons

# %%
max_pred_kurt = 0
max_supp_kurt = 0
for ln in pred_indices:
    if kurtoses[ln[0],ln[1]]>max_pred_kurt:
        best_pred_index = ln
        max_pred_kurt = kurtoses[ln[0],ln[1]]
for ln in supp_indices:
    if kurtoses[ln[0],ln[1]]>max_supp_kurt:
        best_supp_index = ln
        max_supp_kurt = kurtoses[ln[0],ln[1]]
print("clearest prediction neuron:", best_pred_index)
print("clearest suppression neuron:", best_supp_index)

# %% [markdown]
# Yields neurons 31.9634 (depletion) and 29.4180 (orthogonal output)

# %%
neuron_list = [(best_pred_index[0],best_pred_index[1]), (best_supp_index[0], best_supp_index[1])]

# %%
pred_classes = utils.category_lists(pred_indices, gatelin, gateout, linout)

# %%
for key, value in pred_classes.items():
    print(key)
    max_kurt = 0
    for layer,neuron in value:
        if kurtoses[layer,neuron]>max_kurt:
            best_index = (layer,neuron)
            max_kurt = kurtoses[layer,neuron]
    print(f"clearest prediction neuron in class {key}: {best_index} with kurtosis {max_kurt}")
    if key not in ["depletion", "orthogonal output"]:
        neuron_list.append(best_index)

# %%
print(neuron_list)

# %% [markdown]
# ### Weight-based analysis

# %%
for ln in neuron_list:
    ans = utils.neuron_analysis(model, ln[0], ln[1])
