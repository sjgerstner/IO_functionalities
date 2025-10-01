"""
Process a Wiki dump to create two text files:
paragraphs.txt contains plain text paragraphs separated by double newlines;
titles.txt contains the corresponding titles in the format:
"Article title. Section title. Subsection title\n".
"""

from argparse import ArgumentParser
import bz2
from os.path import exists
from tqdm import tqdm

from defusedxml import ElementTree
from wikiextractor.extract import Extractor

parser = ArgumentParser()
parser.add_argument('--scratch_dir')
parser.add_argument(
    '--dump_id',
    default='enwiki-20211020-pages-articles-multistream',
    help="""
    Geva et al. use the Wikipedia dump of October 13, 2021.
    The closest we could find was from October 20, 2021.
    List: https://meta.wikimedia.org/wiki/Data_dump_torrents
    or https://archive.org/search?query=subject%3A%22enwiki%22+AND+subject%3A%22data+dumps%22+AND+collection%3A%22wikimediadownloads%22&sort=-date
    """,
)
args = parser.parse_args()

#multistream (with index)
#index has entries of the type offset:page_id:page_name
#like 597:20460173:Sayf al-Din Ghazi II
#Usually several articles in a row have the same offset,
# you still have to find them within that block.
INDEX_PATH = f'{args.scratch_dir}/{args.dump_id}-index.txt.bz2'
CONTENT_PATH = f'{args.scratch_dir}/{args.dump_id}.xml.bz2'

PARAGRAPHS_PATH = f'{args.scratch_dir}/paragraphs.txt'
TITLES_PATH = f'{args.scratch_dir}/titles.txt'
ID_PATH = f'{args.scratch_dir}/last_processed_id.txt'
OFFSETS_PATH = f'{args.scratch_dir}/offsets.txt'

if exists(ID_PATH):
    with open(ID_PATH, 'r', encoding='utf-8') as f:
        i_start, j_start, k_start = tuple(f.read().split())
else:
    i_start, j_start, k_start = 0,0,0
    for path in [PARAGRAPHS_PATH, TITLES_PATH, ID_PATH]:
        with open(path, 'w', encoding='utf-8') as f:
            f.write('')

#list offsets
if exists(OFFSETS_PATH):
    with open(OFFSETS_PATH, 'r', encoding='utf-8') as f:
        offsets = [int(offset) for offset in f.readlines()]
else:
    f = open(OFFSETS_PATH, 'x', encoding='utf-8')
    f.close()
    offsets = []
    with bz2.open(INDEX_PATH, 'rt', encoding='utf-8') as index:
        for paragraph in index:
            offset = int(paragraph.split(':')[0])
            if (not offsets) or (offsets[-1]!=offset):
                offsets.append(offset)
                with open(OFFSETS_PATH, 'a', encoding='utf-8') as f:
                    print(offset, file=f)

N_STREAMS = len(offsets)
print("Number of streams (?):", N_STREAMS)

extractor = Extractor(id=None, revid=None, urlbase=None, title="", page=None)

starting=True
with open(CONTENT_PATH, 'rb') as file:
    for i, offset in tqdm(enumerate(offsets)):
        if i<i_start:
            continue
        if i==i_start+1:
            starting=False

        file.seek(offset)
        if i+1<len(offsets):
            data = file.read(offsets[i+1]-offset)
        else:
            data = file.read()

        new_data = bz2.decompress(data)
        xml_text = new_data.decode('utf-8')
        #The actual articles thing has xml tags <page><revision></revision></page>
        #and within that we care about <title></title> and <text></text>
        page_xmls = xml_text.split('</page>')[:-1]

        for j, page in enumerate(page_xmls):
            if starting:
                if j<j_start:
                    continue
                if j==j_start+1:
                    starting=False

            page += '</page>'
            #print(page)
            page_tree = ElementTree.fromstring(page)
            title = page_tree.find('title').text
            if ':' in title:
                continue
            text = page_tree.find('revision').find('text').text
            if text is None:
                #print('skipping:', title)
                print(page)
                continue
            #print('processing:', title)
            page_paragraphs = text.split('\n\n')
            #get rid of wiki markup! (but remember section titles)
            current_depth = 1
            current_title = [title]
            for k, paragraph in enumerate(page_paragraphs):
                if starting:
                    if k<k_start:
                        continue
                    if k==k_start:
                        starting=False

                #section title
                if paragraph.startswith('='):
                    split_paragraph = paragraph.split('\n', 1)
                    split0 = split_paragraph[0]
                    #== Section Title ==, === Subsection ===
                    current_depth = split0.count('=')//2
                    current_title = current_title[:current_depth-1] + [split0.strip(' =\n')]
                    if len(split_paragraph)==2:
                        paragraph = split_paragraph[1]
                    else:
                        continue
                #all the rest
                paragraph = ' '.join(
                    extractor.clean_text(paragraph)
                ).strip(
                ).replace(
                    '\n', ' '#if there's a newline in the middle replace it with a space
                )+'\n\n'#previously \n, this is a correction to avoid having to use wiki_clean.py
                if paragraph=='\n\n':#same correction as above
                    continue
                title_to_write = ' '.join(
                    extractor.clean_text('. '.join(current_title))
                ).strip()+'\n'
                if title_to_write=='\n':
                    title_to_write='[UNK]\n'#unknown
                try:
                    with open(PARAGRAPHS_PATH, 'a', encoding='utf-8') as paragraphs_file:
                        print(paragraph, file=paragraphs_file)
                    with open(TITLES_PATH, 'a', encoding='utf-8') as titles_file:
                        print(title_to_write, file=titles_file)
                except UnicodeEncodeError as e:
                    print("Failed to encode Unicode:", title_to_write)
                    continue
                with open(ID_PATH, "w", encoding='utf-8') as id_file:
                    id_file.write(f'{i} {j} {k}')#stream page paragraph
