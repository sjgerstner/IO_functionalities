from argparse import ArgumentParser
import os
from tqdm import tqdm

import bm25s
import pandas as pd

#inspired by
#https://stackoverflow.com/questions/4842057/easiest-way-to-ignore-blank-lines-when-reading-a-file-in-python
def nonblank_lines(file):
    lines = []
    for l in file:
        line = l.rstrip()
        if line:
            lines.append(line)
    return lines

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--knowns_dir')
    parser.add_argument('--wiki_dir')
    parser.add_argument('--out_dir')
    args=parser.parse_args()

    #get list of subjects
    knowns_df = pd.read_json(f'{args.knowns_dir}/known_1000.json')
    subjects = knowns_df['subject'].unique()

    tokenizer = bm25s.tokenization.Tokenizer()
    tokenized_subjects = tokenizer.tokenize(subjects)

    #get corpus and page/section titles
    #we saved them as two txt files
    with open(f'{args.wiki_dir}/paragraphs.txt', 'r', encoding='utf8') as f:
        all_pars = nonblank_lines(f)
        print(len(all_pars), 'paragraphs')
    with open(f'{args.wiki_dir}/titles.txt', 'r', encoding='utf8') as f:
        all_titles = nonblank_lines(f)
        print(len(all_titles), 'titles')
    assert len(all_pars)==len(all_titles)
    n_pars = len(all_titles)
    print(n_pars)

    if not os.path.exists(f'{args.wiki_dir}/retriever'):
        retriever = bm25s.BM25(corpus=all_pars, method="lucene", backend="numba")
        print("tokenizing corpus...")
        corpus_tokens = tokenizer.tokenize(all_pars, update_vocab=False)
        corpus = (corpus_tokens, tokenizer.word_to_id)
        print("indexing...")#we only care about the tokens actually occurring in the subject
        retriever.index(corpus)
        retriever.save(f'{args.wiki_dir}/retriever')
    else:
        retriever = bm25s.BM25.load(f'{args.wiki_dir}/retriever', mmap=True)

    # Given a subject s, first we use the BM25 algorithm
    # (Robertson et al., 1995) to retrieve 100 paragraphs
    # from the English Wikipedia with s being the query.
    print("retrieving...")
    results = retriever.retrieve(
        tokenized_subjects,
        k=100,
        return_as="documents",
        corpus=range(n_pars)
    )
    #returns a numpy array
    #each row corresponds to a subject and contains 100 paragraph indices

    # From the resulting set, we keep only paragraphs for
    # which the subject appears as-is in their content or
    # in the title of the page/section they are in.
    selected_paragraphs=[]
    for s, subject in enumerate(subjects):
        print(subject)
        subject_capitalized = subject.capitalize()
        paragraphs = ''
        for p in tqdm(range(100)):
            i = results[s,p]
            text_to_check = all_titles[i] + '. ' + all_pars[i]
            if subject in text_to_check or subject_capitalized in text_to_check:
                paragraphs = paragraphs + ' ' + all_pars[i]
        # concatenate strings
        selected_paragraphs.append(paragraphs)

    # 2 columns and a header of column names "subject" and "paragraphs".
    # Each entry should have (a) a subject (string) from the "knowns" data (knowns_df)
    # and (b) paragraphs concatenated with space about the subject (a single string).
    df_wiki = pd.DataFrame({'subject':subjects, 'paragraphs':selected_paragraphs})

    df_wiki.to_csv(f'{args.out_dir}/wiki.csv')
