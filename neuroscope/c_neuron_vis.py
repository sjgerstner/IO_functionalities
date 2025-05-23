from argparse import ArgumentParser
import os
import pickle
from tqdm import tqdm

from datasets import load_from_disk
from transformers import AutoTokenizer

from circuitsvis.tokens import colored_tokens

def _vis_example(i, indices, acts, stop_tokens, dataset, tokenizer):
    index = int(indices[i])
    #print(dataset[index]['input_ids'])#tensor of ints
    tokens = tokenizer.batch_decode(#or convert_ids_to_tokens
        dataset[index]['input_ids']
    )[:stop_tokens[i]]
    return f'<h4>Example {i}</h4>\n'+str(
        colored_tokens(
            tokens,
            acts[i][:len(tokens)]
        )
    )+"\n</div>"

def neuron_vis_full(neuron_data, dataset, tokenizer):
    htmls = []
    # # We first add the style to make each token element have a nice border
    # htmls = [style_string]
    # We then add a line telling us the limits of our range
    htmls.append(
        f"""<p>
        Max Act.: <b>{neuron_data['max_val']:.4f}</b>.
        Min Act.: <b>{neuron_data['min_val']:.4f}</b>.
        Avg Act.: <b>{neuron_data['avg_val']:.4f}</b>.
        </p>
        <p>
        Act. Frequency: <b>{neuron_data['act_freq']:.2%}</b>.
        </p>"""
    )
    htmls.append('<h3>Max activations</h3>')
    for i in range(neuron_data['max_indices'].shape[0]):
        # print(max_indices[i])
        # print(dataset[int(max_indices[i])])
        htmls.append(
            _vis_example(
                i,
                neuron_data['max_indices'],
                neuron_data['max_acts'],
                neuron_data['argmax_tokens'],
                dataset,
                tokenizer
                )
            )
    htmls.append('<hr>')
    htmls.append('<h3>Min activations</h3>')
    for i in range(neuron_data['min_indices'].shape[0]):
        htmls.append(
            _vis_example(
                i,
                neuron_data['min_indices'],
                neuron_data['min_acts'],
                neuron_data['argmin_tokens'],
                dataset,
                tokenizer
                )
            )
    return "\n".join(htmls)

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument('--datadir', default=None)
    parser.add_argument('--model', default='allenai/OLMo-1B-hf')
    parser.add_argument('--dataset',
                        default='dolma_small',
                        help='pretokenized dataset'
                        )
    parser.add_argument('--save_to',
                        default=None,
                        help='should be the same as the corresponding b_activations.py run.'
                        )
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    if args.save_to:
        RUN_CODE = args.save_to
    elif args.test:
        RUN_CODE = "test"
    else:
        RUN_CODE = f"{args.model.split('/')[-1]}_{args.dataset.split('-')[0]}"
    #OLMO-1B-hf_dolma-v1_7-3B
    #the id of the b-activations.py run

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = load_from_disk(f'{args.datadir}/{args.dataset}')

    with open(f'{args.datadir}/{RUN_CODE}/summary.pickle', 'rb') as f:
        summary_dict = pickle.load(f)
        print(f"summary_dict: {summary_dict.keys()}")
    with open(f'{args.datadir}/{RUN_CODE}/indices.pickle', 'rb') as f:
        maxmin_indices = pickle.load(f) #sample layer neuron
        print(f'maxmin_indices: {maxmin_indices.shape}')
    with open(f'{args.datadir}/{RUN_CODE}/activations.pickle', 'rb') as f:
        activations_dict = pickle.load(f)
        print(f"activations_dict['acts']: {activations_dict['acts'].shape}")

    TOPK = activations_dict['acts'].shape[2]//2
    if args.test:
        NLAYERS=1
        NNEURONS=1
    else:
        NLAYERS=activations_dict['acts'].shape[0]
        NNEURONS=activations_dict['acts'].shape[1]

    TITLE = f"<h1>Model: <b>{args.model}</b></h1>\n"

    for layer in range(NLAYERS):
        if not os.path.exists(f"{args.datadir}/{RUN_CODE}/L{layer}"):
            os.mkdir(f"{args.datadir}/{RUN_CODE}/L{layer}")
        for neuron in tqdm(range(NNEURONS)):
            neuron_data = {
                'max_indices':maxmin_indices[:TOPK, layer, neuron],
                'min_indices':maxmin_indices[TOPK:, layer, neuron],
                'max_acts':activations_dict['acts'][layer,neuron, :TOPK],
                'min_acts':activations_dict['acts'][layer,neuron, TOPK:],
                'max_val':summary_dict['summary_max'][layer,neuron],
                'min_val':summary_dict['summary_min'][layer,neuron],
                'avg_val':summary_dict['summary_mean'][layer,neuron],
                'act_freq':summary_dict['summary_freq'][layer,neuron],
                'argmax_tokens':summary_dict['argmax_activations'][:,layer,neuron],
                'argmin_tokens':summary_dict['argmin_activations'][:,layer,neuron],
            }
            # We add some text to tell us what layer and neuron we're looking at
            heading = f"<h2>Layer: <b>{layer}</b>. Neuron Index: <b>{neuron}</b></h2>\n"
            HTML = TITLE + heading + neuron_vis_full(
                   neuron_data=neuron_data,
                   dataset=dataset,
                   tokenizer=tokenizer,
            )
            with open(
                f'{args.datadir}/{RUN_CODE}/L{layer}/N{neuron}.html',
                'w',
                encoding="utf-8",
            ) as f:
                f.write(HTML)
    print('done!')
