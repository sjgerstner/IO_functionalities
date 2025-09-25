"""Check if models have weight tying"""
from gc import collect
from os.path import exists

import torch
from transformer_lens import HookedTransformer

from main import MODEL_LIST

def check_weight_tying(model_name, **kwargs):
    """Load given model and check if it has weight tying (True) or not (False)"""
    model = HookedTransformer.from_pretrained_no_processing(
        model_name,
        #local_files_only=True,#leads to error with mistral
        **kwargs,
    )

    ans = torch.all(model.W_E.T == model.W_U)

    del model
    collect()
    torch.cuda.empty_cache()

    return ans

if __name__=="__main__":
    PATH = "weight_tying.txt"
    if exists(PATH):
        with open(PATH, 'r', encoding='utf-8') as f:
            N = len(f.readlines())
    else:
        N=0
    for model_name in MODEL_LIST[N:]:
        wt = check_weight_tying(model_name)
        with open(PATH, "a", encoding="utf-8") as f:
            if wt:
                f.write(f"{model_name} has weight tying.\n")
            else:
                f.write(f"{model_name} does not have weight tying.\n")
