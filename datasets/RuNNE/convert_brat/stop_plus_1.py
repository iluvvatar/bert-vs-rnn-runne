import os
import json
from pathlib import Path


home_dir = Path(os.getenv('HOME'))
dataset_dir = home_dir / 'datasets' / 'RuNNE'
files_from = [
    dataset_dir / 'data_(stop=last_symbol)/train.jsonl',
    dataset_dir / 'data_(stop=last_symbol)/test.jsonl',
    dataset_dir / 'data_(stop=last_symbol)/dev.jsonl'
]
files_to = [
    dataset_dir / 'HuggingFaceHub/data/train.jsonl',
    dataset_dir / 'HuggingFaceHub/data/test.jsonl',
    dataset_dir / 'HuggingFaceHub/data/dev.jsonl'
]
for file_in, file_out in zip(files_from, files_to):
    with open(file_in, encoding='utf-8') as f_in, open(file_out, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            line = json.loads(line)
            # print(line)
            entities = [e.split() for e in line['entities']]
            entities = [f'{start} {int(stop) + 1} {type_}' for start, stop, type_ in entities]
            line['entities'] = entities
            line = json.dumps(line, ensure_ascii=False)
            print(line, file=f_out)
