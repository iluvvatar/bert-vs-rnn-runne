# =============================================================================
# Convert brat format dataset from https://github.com/nerel-ds/NEREL to jsonl
# =============================================================================

from pathlib import Path
from typing import Union
import os
import json
import re

from mlpipeline.datasets.nlp.units import Relation, Entity, Link
from mlpipeline.utils import spaces_pattern


def ann_files(folder_path: Union[str, Path]):
    for root, dirs, files in os.walk(folder_path):
        root = Path(root)
        for file in files:
            if file.endswith('.ann'):
                yield root / Path(file)


def brat_to_dict(ann_file_path: Union[str, Path],
                 doc_id: int):
    doc = {'id': doc_id}
    if not isinstance(ann_file_path, Path):
        ann_file_path = Path(ann_file_path)
    assert ann_file_path.suffix == '.ann'
    with open(ann_file_path.with_suffix('.txt'), encoding='utf-8') as f_txt:
        doc['text'] = re.sub(spaces_pattern, ' ', f_txt.read())
    doc['entities'] = []
    doc['relations'] = []
    doc['links'] = []
    with open(ann_file_path, encoding='utf-8') as f_ann:
        for line in f_ann:
            line = re.sub(spaces_pattern, ' ', line.strip('\n'))
            try:
                if line.startswith('T'):
                    doc['entities'].append(Entity.from_brat(line))
                elif line.startswith('R'):
                    doc['relations'].append(Relation.from_brat(line))
                elif line.startswith('N'):
                    doc['links'].append(Link.from_brat(line))
                else:
                    raise Exception
            except Exception as exc:
                raise ValueError(f'Unknown format: {line}\n'
                                 f'original exception: {exc}\n'
                                 f'file: {ann_file_path}')
    doc['entities'] = list(map(str, sorted(doc['entities'], key=lambda x: x.id)))
    doc['relations'] = list(map(str, sorted(doc['relations'], key=lambda x: x.id)))
    doc['links'] = list(map(str, sorted(doc['links'], key=lambda x: x.id)))
    return doc


def brat_folder_to_jsonl(folder_path: Union[str, Path],
                         file_out_path: Union[str, Path]):
    with open(file_out_path, 'w', encoding='utf-8') as f_out:
        for i, ann_file in enumerate(ann_files(folder_path)):
            doc = brat_to_dict(ann_file, doc_id=i)
            print(json.dumps(doc, ensure_ascii=False), file=f_out)


# Annotation config
# ============================================================================

def ent_types_tsv_to_jsonl(file_in_path: Union[str, Path],
                           file_out_path: Union[str, Path]):
    with open(file_in_path) as f_in, open(file_out_path, 'w') as f_out:
        for line in f_in:
            line = re.sub(spaces_pattern, ' ', line).strip().split('\t')
            line = {
                'type': line[0],
                'link': line[1] if len(line) == 2 else ''
            }
            print(json.dumps(line), file=f_out)


def rel_types_tsv_to_jsonl(file_in_path: Union[str, Path],
                           file_out_path: Union[str, Path]):
    with open(file_in_path) as f_in, open(file_out_path, 'w') as f_out:
        for line in f_in:
            type_, args = re.sub(spaces_pattern, ' ', line).strip().split('\t')
            arg1, arg2 = args.split(', ')
            arg1 = arg1[5:].split('|')
            arg2 = arg2[5:].split('|')
            line = {
                'type': type_,
                'arg1': arg1,
                'arg2': arg2
            }
            print(json.dumps(line), file=f_out)


if __name__ == '__main__':
    home_dir = Path(os.getenv('HOME'))
    root_folder = home_dir / 'datasets' / 'NEREL'
    data_root_folder = root_folder / 'NEREL-v1.1'
    hf_root_folder = root_folder / 'HuggingFaceHub'
    brat_folder_to_jsonl(data_root_folder / 'dev', hf_root_folder / 'data' / 'dev.jsonl')
    brat_folder_to_jsonl(data_root_folder / 'train', hf_root_folder / 'data' / 'train.jsonl')
    brat_folder_to_jsonl(data_root_folder / 'test', hf_root_folder / 'data' / 'test.jsonl')
    ent_types_tsv_to_jsonl(root_folder / 'ent_types.tsv',
                           hf_root_folder / 'ent_types.jsonl')
    rel_types_tsv_to_jsonl(root_folder / 'rel_types.tsv',
                           hf_root_folder / 'rel_types.jsonl')
