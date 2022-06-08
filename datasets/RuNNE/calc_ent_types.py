from pathlib import Path

from mlpipeline.datasets.nlp import RuNNE
from mlpipeline.datasets.nlp.units import Entity

from tqdm import tqdm


# 'AGE': 692,
# 'AWARD': 457,
# 'CITY': 1342,
# 'COUNTRY': 2977,
# 'CRIME': 217,
# 'DATE': 2807,
# 'DISEASE': 89,
# 'DISTRICT': 123,
# 'EVENT': 3600,
# 'FACILITY': 434,
# 'FAMILY': 31,
# 'IDEOLOGY': 343,
# 'LANGUAGE': 51,
# 'LAW': 458,
# 'LOCATION': 336,
# 'MONEY': 214,
# 'NATIONALITY': 460,
# 'NUMBER': 1256,
# 'ORDINAL': 672,
# 'ORGANIZATION': 4744,
# 'PENALTY': 51,
# 'PERCENT': 89,
# 'PERSON': 5480,
# 'PRODUCT': 291,
# 'PROFESSION': 5480,
# 'RELIGION': 118,
# 'STATE_OR_PROVINCE': 455,
# 'TIME': 201,
# 'WORK_OF_ART': 127}

def main():
    home_dir = Path('D:/Research/RNNvsBERT')
    runne_path = home_dir / 'datasets' / 'RuNNE' / 'Preprocessed'

    runne = RuNNE.load(runne_path)
    counts = {ent_type: 0 for ent_type in runne.entity_types}
    for split in runne:
        dataset = runne[split]
        for example in tqdm(dataset):
            for e in example['entities']:
                e = Entity.from_str(e)
                counts[e.type] += 1
    print(counts)
    print(sorted([(ent_type, count) for ent_type, count in counts.items()],
                  key=lambda x: x[1]))


if __name__ == '__main__':
    main()
