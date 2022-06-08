# =============================================================================
# DON'T USE THIS WEIGHTS!!!
# This way doesn't work!
# To use weights calc them for each entity type separately
# =============================================================================
import os
from pathlib import Path
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from mlpipeline.datasets.nlp import RuNNE
from mlpipeline.datasets.nlp.samplers import UnbalancedEntitiesSampler
from mlpipeline.datasets.nlp.units import Entity


from tqdm import tqdm

# train_counts = {'O': 5566713, 'B': 28065, 'L': 28065, 'I': 39323, 'U': 13540}
# test_counts = {'O': 1270869, 'B': 6589, 'L': 6589, 'I': 9111, 'U': 3316}
# train_weights = np.zeros(5)
# test_weights = np.zeros(5)
# for key, idx in BILOULabelizer.bilou2int.items():
#     train_weights[idx] = 1 / train_counts[key]
#     test_weights[idx] = 1 / test_counts[key]
#     # train_weights[idx] = train_counts[key]
#     # test_weights[idx] = test_counts[key]
# weights = train_weights + test_weights
# weights /= weights.sum()
# print(weights)
weights_train = np.array([0.20364551, 44.13734303, 44.13734303, 33.49500502, 69.74517233])
weights_test = np.array([0.20370485, 43.48096953, 43.48096953, 33.14366653, 67.71403207])
weights = (weights_train + weights_test) / 2
# print(weights)

def main():
    home_dir = Path(os.getenv('HOME'))
    runne_path = home_dir / 'datasets' / 'RuNNE' / 'Preprocessed'

    runne = RuNNE.load(runne_path)
    for split in runne:
        dataset = runne[split]
        sampler = UnbalancedEntitiesSampler(dataset,
                                            entities_deserialize_fn=Entity.from_str,
                                            entities_column='entities',
                                            tokens_spans_column='spans',
                                            entity_types_shares='log',
                                            size=10_000)
        y = []
        for index in tqdm(sampler):
            example = dataset[index]
            for lab in example['labels_ids']:
                y.extend(lab)
        weights = compute_class_weight('balanced', classes=np.arange(5), y=y)
        # weights = {key: 0 for key in BILOULabelizer.bilou2int}
        # for index in tqdm(sampler):
        #     example = dataset[index]
        #     labels_ids = example['labels_ids']
        #     for ent_type_labels_ids in labels_ids:
        #         for label_id in ent_type_labels_ids:
        #             key = BILOULabelizer.int2bilou[label_id]
        #             weights[key] += 1
        print(split, weights)
        # train {'O': 5566713, 'B': 28065, 'L': 28065, 'I': 39323, 'U': 13540}
        # test {'O': 1270869, 'B': 6589, 'L': 6589, 'I': 9111, 'U': 3316}
        # dev ValueError: No one entities provided in dataset.


if __name__ == '__main__':
    main()
