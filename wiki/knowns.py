from argparse import ArgumentParser
#import re
from tqdm import tqdm
#import wget

import pandas as pd

from transformer_lens import HookedTransformer

parser = ArgumentParser()
parser.add_argument('--model', default='allenai/OLMo-7B-0424-hf')
parser.add_argument('--data_dir')
args = parser.parse_args()

# Get CounterFact data for GPT2-xl, from the ROME repository.
#wget.download("https://rome.baulab.info/data/dsets/known_1000.json")
knowns_df = pd.read_json(f"{args.data_dir}/known_1000.json")

# Load model
model = HookedTransformer.from_pretrained(args.model)
eos_id = model.tokenizer.eos_token_id

#examples which OLMo knows
#from Meng et al, ROME paper:
# "we perform greedy generation using facts and fact
# templates from COUNTERFACT, and we identify predicted text that names the correct object oc
# before naming any other capitalized word.
# We use the text up to but not including the object oc as the
# prompt, and we randomly sample 1000 of these texts"

count_failures = 0
for index, row in tqdm(knowns_df.iterrows()):
    attribute_len = len(row['attribute'])
    prompt = row['template'].replace('{}', row['subject']).rstrip()
    if prompt.startswith(row['subject']):
        prompt = prompt[0].upper() + prompt[1:]
        #don't use .capitalize(), it will lowercase the other chars!
    prompt = ' ' + prompt
    whole_str = prompt
    prompt_len = len(prompt)
    previous_len = len(whole_str)
    upper_found=False
    success = False
    possible_success=False
    failure = False
    count_possible_successes = 0
    while not success and not failure:# and not possible_success:
        if not possible_success:
            previous_len = len(whole_str)
        # else:
        #     count_possible_successes+=1
        #     print(whole_str)
        #     # if count_possible_successes==2:
        #     #     break
        all_tokens = model.generate(
            whole_str, max_new_tokens=10, top_k=1, return_type='tokens'
        )[0,1:]#without EOS
        whole_str = model.to_string(all_tokens)
        answer_str = whole_str[prompt_len:]
        new_str = whole_str[previous_len:]
        new_str_split = new_str.split()
        for i,word in enumerate(new_str_split):
            if word[0].isupper():
                upper_found=True
                if row['attribute'].lower() in (answer_str.split(word)[0]+word).lower():
                    success=True
                elif row['attribute'].lower().startswith(word.lower()):
                    str_to_search = ' '.join(new_str_split)[i:]
                    if len(str_to_search)>=attribute_len or eos_id in all_tokens:
                        if str_to_search.lower().startswith(row['attribute'].lower()):
                            success=True
                        else:
                            failure=True
                    else:
                        possible_success=True
                        print('Possible success:', whole_str)
                else:
                    failure=True
                break
        if (not upper_found) and (eos_id in all_tokens):
            if row['attribute'].lower() in answer_str.lower():
                success=True
            else:
                failure=True
    if failure:# or possible_success:
        count_failures+=1
        print(whole_str)
        print(row['attribute'])
        knowns_df = knowns_df.drop(index=index)
        # if count_failures==6:
        #     break
    else:
        assert success
        #which version of the attribute did the model generate?
        delimiters = row['attribute'], row['attribute'].lower(), row['attribute'].capitalize()
        delimiter=None
        for d in delimiters:
            if d in whole_str:
                delimiter = d
                break
        assert delimiter is not None
        prompt_and_prediction = whole_str.split(delimiter)
        final_prompt = prompt_and_prediction[0]
        prediction = delimiter + delimiter.join(prompt_and_prediction[1:])
        knowns_df.loc[knowns_df['known_id']==index, 'prompt'] = final_prompt
        knowns_df.loc[knowns_df['known_id']==index, 'prediction'] = prediction

print('number of failures:', count_failures)

knowns_df.to_json(f'{args.data_dir}/known_{args.model.split('/')[-1]}.json', orient='records')
