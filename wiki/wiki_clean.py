from tqdm import tqdm

from wiki_retrieve import nonblank_lines

with open('/mounts/work/sgerstner/wiki/paragraphs.txt', 'r+', encoding='utf-8') as f:
    lines = nonblank_lines(f)
    for i,_line in tqdm(enumerate(lines)):
        lines[i] = lines[i].replace('\n', '\n\n')
        lines[i]+='\n'
    f.writelines(lines)
