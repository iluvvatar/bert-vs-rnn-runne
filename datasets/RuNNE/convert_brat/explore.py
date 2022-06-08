import os
import json
from pathlib import Path
from mlpipeline.datasets.nlp.units import Entity


def main(path: Path):
    with open(path, encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            entities = [Entity.from_str(e) for e in doc['entities']]
            entities = [(e.type, e.start, e.stop, doc['text'][e.start:e.stop]) for e in entities]
            print(doc['id'], entities)


if __name__ == '__main__':
    home_dir = Path(os.getenv('HOME'))
    dataset_dir = home_dir / 'datasets' / 'RuNNE'
    main(dataset_dir / 'HuggingFaceHub/data/train.jsonl')
