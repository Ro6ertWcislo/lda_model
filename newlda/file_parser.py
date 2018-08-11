import json

import os

from codecs import open


def parse_wiki_dump(filename, limit=None):
    texts = []
    buffer = []
    with open(filename, "r", encoding='utf-8', errors='replace') as f:
        for line in f.readlines():
            if line.startswith(" =") and line.endswith("= \n") and line.count('=') == 2:
                if buffer:
                    texts.append(''.join(buffer))
                    buffer = []
                    if limit and limit < len(texts):
                        return texts[1:]
            trimmed_line = line.replace('<unk>', '')
            buffer.append(trimmed_line)
        if buffer:
            texts.append(''.join(buffer))
    return texts[1:]


def parse_dir_json(dirpath, limit=None):
    documents = []
    for file in os.listdir(dirpath):
        with open(dirpath + '/' + file, "r", encoding='utf-8', errors='replace') as f:
            data = json.load(f)
            content = data['title'] + ' ' + ' '.join(data['author']) + ' ' + data['description'] + ' ' + data['content']
            documents.append((data['url'], content))
        if limit is not None and len(documents) > limit:
            return documents
    return documents
