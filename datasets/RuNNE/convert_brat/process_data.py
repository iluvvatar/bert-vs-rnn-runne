import json
import re
import os
from pathlib import Path


from mlpipeline.utils import spaces_pattern


def process_data_file(file_in, file_out):
    docs = {}   # id: dict
    with open(file_in, encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            doc_id = doc['id']
            text = re.sub(spaces_pattern, ' ', doc['sentences'])
            if 'ners' in doc:
                entities = [f'{start} {int(stop)+1} {type_}' for start, stop, type_ in doc['ners']]
                docs[doc_id] = {'id': doc_id,
                                'text': text,
                                'entities': entities}
            else:
                docs[doc_id] = {'id': doc_id,
                                'text': text,
                                'entities': []}
    ids = sorted(docs.keys())
    with open(file_out, 'w', encoding='utf-8') as f:
        for doc_id in ids:
            doc = docs[doc_id]
            print(json.dumps(doc, ensure_ascii=False), file=f)


if __name__ == '__main__':
    home_dir = Path(os.getenv('HOME'))
    dataset_dir = home_dir / 'datasets' / 'RuNNE'
    process_data_file(dataset_dir / '_dev.jsonl', dataset_dir / 'HuggingFaceHub/data/dev.jsonl')
    process_data_file(dataset_dir / '_train.jsonl', dataset_dir / 'HuggingFaceHub/data/train.jsonl')
    process_data_file(dataset_dir / '_test.jsonl', dataset_dir / 'HuggingFaceHub/data/test.jsonl')
